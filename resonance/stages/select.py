"""
select: ClipManifest -> ClipManifest. Keeps a bounded number of clips per speaker.

Simple rules for now: a speaker with fewer than min_per_speaker clips is left out; one with more than
max_per_speaker clips, or more than max_minutes_per_speaker of audio, keeps a random subset within both caps.
"""

import collections
import random

from resonance.config.schema import SelectConfig
from resonance.io import manifest
from resonance.io.manifest import Row
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage


def choose(rows: list[Row], config: SelectConfig) -> list[Row]:
    """
    The selected rows, in their original order.

    Each speaker's clips are shuffled with a seed made of config.seed and the speaker ID, so a speaker's selection
    does not change when other speakers join or leave the pool.
    """

    by_speaker = collections.defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)
    limit = None if config.max_minutes_per_speaker is None else config.max_minutes_per_speaker * 60
    chosen = set()
    for speaker, clips in by_speaker.items():
        if len(clips) < config.min_per_speaker:
            continue
        order = sorted(clips, key=lambda row: row["name"])
        random.Random(f"{config.seed}:{speaker}").shuffle(order)
        taken, seconds = 0, 0.0
        for row in order:
            duration = float(row["duration"])
            if taken == config.max_per_speaker:
                break
            if limit is not None and seconds + duration > limit:
                continue
            chosen.add(row["name"])
            taken, seconds = taken + 1, seconds + duration
    return [row for row in rows if row["name"] in chosen]


class Select(Stage):
    name = "select"
    scope = "pool"
    consumes = ClipManifest
    produces = ClipManifest

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "select.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        rows = manifest.read(source.path)
        selected = choose(rows, self.config.select)
        manifest.write(ctx.work_dir / "select.csv", selected)
        ctx.summary.update(clips_in=len(rows), clips_out=len(selected),
                           hours_out=round(manifest.hours(selected), 3),
                           speakers_in=len({row["speaker_id"] for row in rows}),
                           speakers_out=len({row["speaker_id"] for row in selected}))
        return self.output(ctx)
