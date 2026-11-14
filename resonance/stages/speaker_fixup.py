"""
speaker_fixup: ClipManifest -> ClipManifest. Merges the source IDs that are one person recorded in several videos.

The corpus carries no speaker field, so speaker_id starts out as the video ID. A channel host appears in every
video they publish, so one person routinely holds dozens of IDs. This embeds a few clips per source, clusters the
sources the model calls one person, and rewrites speaker_id to the lowest ID of each cluster. No other column
changes. merges.csv lists every source that moved. Run it over a whole shard or several: a small sample of
recordings rarely shares a host.
"""

import collections

from resonance.io import manifest
from resonance.pipeline import gpu
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage
from resonance.speakers import clustering, embedding

MERGE_COLUMNS = ("speaker_id", "merged_into", "min_agreement", "closest_clip_pair", "duplicate_audio")


class SpeakerFixup(Stage):
    name = "speaker_fixup"
    scope = "pool"
    consumes = ClipManifest
    produces = ClipManifest
    needs_gpu = True

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "speaker_fixup.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config = self.config.speaker_fixup
        rows = manifest.read(source.path)
        clips = collections.defaultdict(list)
        for row in rows:
            clips[row["speaker_id"]].append(ctx.layout.root / row["audio_path"])
        sources = [(speaker, embedding.longest(paths, config.clips_per_speaker))
                   for speaker, paths in sorted(clips.items())]

        embeddings: dict = {}
        for part in gpu.map_gpus(embedding.embed, gpu.split(sources, len(ctx.gpus)), ctx.gpus,
                                 self.config.resources.threads_per_worker, ctx.log.progress(len(sources)),
                                 self.log_path(ctx)):
            embeddings.update(part)

        agreement, closest = clustering.compare(embeddings)
        order = sorted(embeddings)
        merges = clustering.merge_map(order, agreement, config.threshold, config.linkage)
        index = {speaker: position for position, speaker in enumerate(order)}
        families = collections.defaultdict(list)
        for speaker, canonical in merges.items():
            families[canonical].append(speaker)
        records = []
        for canonical, members in sorted(families.items()):
            group = [index[speaker] for speaker in [canonical, *members]]
            pairs = [(a, b) for position, a in enumerate(group) for b in group[position + 1:]]
            nearest = max(float(closest[a, b]) for a, b in pairs)
            records += [{"speaker_id": member, "merged_into": canonical,
                         "min_agreement": f"{min(agreement[a, b] for a, b in pairs):.3f}",
                         "closest_clip_pair": f"{nearest:.3f}",
                         "duplicate_audio": str(nearest >= clustering.DUPLICATE)} for member in sorted(members)]
        for row in rows:
            row["speaker_id"] = merges.get(row["speaker_id"], row["speaker_id"])

        manifest.write(ctx.work_dir / "merges.csv", records, MERGE_COLUMNS)
        manifest.write(ctx.work_dir / "speaker_fixup.csv", rows)
        ctx.summary.update(clips_out=len(rows), hours_out=round(manifest.hours(rows), 3),
                           speakers_in=len(clips), speakers_out=len(clips) - len(merges),
                           largest_cluster=max((len(members) + 1 for members in families.values()), default=1))
        return self.output(ctx)
