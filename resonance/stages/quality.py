"""quality: ClipManifest -> ClipManifest + verdicts.csv. Judges every clip with the quality cascade.

Each worker of a spawned pool holds every model on the GPU, reads its clip, runs the cascade and sends back only
the verdict and the scores. verdicts.csv lists every input clip with its verdict and the scores behind it, so a
threshold can be re-tuned, and the source pass can count flags, without running a model again.
"""

import collections
import json
import warnings

from resonance.config.schema import QualityConfig
from resonance.io import manifest
from resonance.io.audio import read_audio
from resonance.metrics.cascade import ACCEPTED, ERROR, Cascade
from resonance.pipeline import pool
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage, guard

VERDICT_COLUMNS = ("name", "source_id", "verdict", "duration", "scores", "error")

_cascade: Cascade | None = None


def _setup(config: QualityConfig) -> None:
    """Every model once per worker. Single-threaded torch, or the processes fight over cores."""

    global _cascade
    import torch

    torch.set_num_threads(1)
    # TF32 matmuls, stated here because every threshold was calibrated with them: the pre-refactor filter imported
    # speechbrain in each worker, and that import switches them on. Without it CTC scores move in the third decimal.
    torch.set_float32_matmul_precision("high")
    warnings.simplefilter("ignore")
    _cascade = Cascade(config)


def _score(task: tuple[str, str]) -> tuple[str, float, dict, str | None]:
    """One clip through the cascade: (verdict, duration, scores, error)."""

    path, transcript = task
    try:
        audio = read_audio(path)
        verdict, scores = _cascade.run(audio, transcript)
        return verdict, len(audio["audio"]) / audio["sample_rate"], scores, None
    except Exception as error:
        return ERROR, 0.0, {}, f"{type(error).__name__}: {error}".replace("\n", " ")


class Quality(Stage):
    name = "quality"
    scope = "batch"
    consumes = ClipManifest
    produces = ClipManifest
    needs_gpu = True
    provides = ("verdicts.csv",)

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "quality.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config = self.config.quality
        rows = manifest.read(source.path)
        tasks = [(str(ctx.layout.root / row["audio_path"]), row["transcription"]) for row in rows]
        progress = ctx.log.progress(len(rows))
        kept, records, counts, errors = [], [], collections.Counter(), 0
        for row, (verdict, duration, scores, error) in zip(
                rows, pool.imap(_score, tasks, config.workers, _setup, (config,))):
            progress.advance()
            if error:
                errors += 1
                ctx.log.error(f"{row['name']}: {error}")
                guard(error, errors, len(rows), config.max_error_rate)
            counts[verdict] += 1
            records.append({"name": row["name"], "source_id": row["source_id"], "verdict": verdict,
                            "duration": f"{duration:.6f}", "scores": json.dumps(scores, default=float),
                            "error": error or ""})
            if verdict == ACCEPTED:
                kept.append(row)
        progress.flush()

        manifest.write(ctx.work_dir / "verdicts.csv", records, VERDICT_COLUMNS)
        manifest.write(ctx.work_dir / "quality.csv", kept)
        ctx.summary.update(clips_in=len(rows), clips_out=len(kept), hours_out=round(manifest.hours(kept), 3),
                           errors=errors, rejected={verdict: count for verdict, count in counts.items()
                                                    if verdict not in (ACCEPTED, ERROR)})
        return self.output(ctx)
