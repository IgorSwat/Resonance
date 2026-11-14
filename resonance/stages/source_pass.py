"""source_pass: ClipManifest + verdicts.csv -> ClipManifest. Drops whole recordings that look like conversations.

A recording whose diarised clips are flagged this often is an interview or a panel, and the rest of its clips
are not trusted either: the per-clip metric cannot see that, because each single clip really is single-voice.
The rate is taken over the clips the multi-speaker metric actually scored, not over every clip: one rejected
earlier in the cascade was never diarised and says nothing about the recording.
"""

import collections

from resonance.config.schema import Config
from resonance.io import manifest
from resonance.metrics.cascade import ACCEPTED
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage

GATE = "multi_speaker"


def flagged_sources(verdicts: list[dict[str, str]], order: list[str], max_flag_rate: float,
                    min_clips: int) -> set[str]:
    """Sources where the multi-speaker metric rejected at least max_flag_rate of the clips it scored.

    `order` is the cascade's metric order: a clip reached the multi-speaker metric when it was accepted or
    rejected by that metric or a later one. Sources with fewer than min_clips scored clips are left alone.
    """

    reached = {ACCEPTED, *order[order.index(GATE):]}
    scored, flagged = collections.Counter(), collections.Counter()
    for record in verdicts:
        if record["verdict"] in reached:
            scored[record["source_id"]] += 1
            flagged[record["source_id"]] += record["verdict"] == GATE
    return {source for source, count in scored.items()
            if count >= min_clips and flagged[source] / count >= max_flag_rate}


class SourcePass(Stage):
    name = "source_pass"
    scope = "batch"
    consumes = ClipManifest
    produces = ClipManifest
    needs = ("verdicts.csv",)

    @classmethod
    def enabled(cls, config: Config) -> bool:
        return config.source_pass.enabled and GATE in config.quality.metrics

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "source_pass.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config = self.config.source_pass
        rows = manifest.read(source.path)
        dropped = flagged_sources(manifest.read(ctx.work_dir / "verdicts.csv"), self.config.quality.metrics,
                                  config.max_flag_rate, config.min_clips)
        kept = [row for row in rows if row["source_id"] not in dropped]
        manifest.write(ctx.work_dir / "source_pass.csv", kept)
        ctx.summary.update(clips_in=len(rows), clips_out=len(kept), hours_out=round(manifest.hours(kept), 3),
                           sources_dropped=len(dropped))
        return self.output(ctx)
