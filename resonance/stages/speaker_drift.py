"""
speaker_drift: ClipManifest -> ClipManifest. Drops clips whose voice changes partway through.

The last guard against a second voice. The cascade's multi-speaker metric asks who speaks when, so it merges two
similar voices taking clean turns; the source pass judges a whole recording, so it cannot remove a good
recording's one bad clip. SpeakerDriftMetric (ECAPA) asks only whether the voice itself changes between two
moments of the clip. Its similarity falls with clip length, so a clip is flagged by z within its duration band,
which needs every clip of the batch scored first. speaker_drift_scores.csv keeps every score.
"""

import warnings

import numpy as np

from resonance.config.schema import Config
from resonance.io import manifest
from resonance.io.audio import read_audio
from resonance.metrics.banding import banded_z
from resonance.pipeline import pool
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage, guard

_metric = None


def _setup() -> None:
    global _metric
    import torch

    from resonance.metrics.speaker_drift import SpeakerDriftMetric

    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    _metric = SpeakerDriftMetric()


def _score(path: str) -> tuple[float | None, str | None]:
    try:
        return _metric.evaluate(read_audio(path))["similarity"], None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}".replace("\n", " ")


def drifting(similarities: np.ndarray, durations: np.ndarray, max_z: float) -> tuple[np.ndarray, np.ndarray]:
    """(z, flagged): each similarity as a z within its duration band, and whether it is at or below max_z."""

    z = banded_z(similarities, durations)
    return z, z <= max_z


class SpeakerDrift(Stage):
    name = "speaker_drift"
    scope = "batch"
    consumes = ClipManifest
    produces = ClipManifest
    needs_gpu = True

    @classmethod
    def enabled(cls, config: Config) -> bool:
        return config.speaker_drift.enabled

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "speaker_drift.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config = self.config.speaker_drift
        rows = manifest.read(source.path)
        progress = ctx.log.progress(len(rows))
        scored, errors = [], 0
        paths = [str(ctx.layout.root / row["audio_path"]) for row in rows]
        for row, (similarity, error) in zip(rows, pool.imap(_score, paths, config.workers, _setup, ())):
            progress.advance()
            if error:
                errors += 1
                ctx.log.error(f"{row['name']}: {error}")
                guard(error, errors, len(rows), self.config.quality.max_error_rate)
            else:
                scored.append((row, similarity))
        progress.flush()

        similarities = np.array([similarity for _, similarity in scored])
        durations = np.array([float(row["duration"]) for row, _ in scored])
        z, flagged = drifting(similarities, durations, config.max_z) if scored else (np.array([]), np.array([]))
        kept = [row for (row, _), drop in zip(scored, flagged) if not drop]
        manifest.write(ctx.work_dir / "speaker_drift_scores.csv",
                       [{"name": row["name"], "similarity": f"{similarity:.6f}", "z": f"{value:.4f}",
                         "flagged": str(bool(drop))} for (row, similarity), value, drop in zip(scored, z, flagged)],
                       ("name", "similarity", "z", "flagged"))
        manifest.write(ctx.work_dir / "speaker_drift.csv", kept)
        ctx.summary.update(clips_in=len(rows), clips_out=len(kept), hours_out=round(manifest.hours(kept), 3),
                           flagged=int(np.sum(flagged)), errors=errors)
        return self.output(ctx)
