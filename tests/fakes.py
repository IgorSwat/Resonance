"""Stand-ins for the GPU stages, so the runner and the real CPU stages can be tested end to end without a GPU."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from resonance.config import loader
from resonance.config.schema import Config
from resonance.io import manifest
from resonance.pipeline.artifacts import BatchRef, ClipManifest, RawBatch
from resonance.pipeline.stage import Context, Stage

CLIPS_PER_SOURCE, SOURCES = 6, 3


def config(tmp_path: Path, first: int = 0, last: int = 1, **overrides: object) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text(f"run: {{name: test, data_root: {tmp_path / 'data'}}}\n"
                    f"source: {{language: en, batches: [{{shard: '000', first: {first}, last: {last}}}]}}\n")
    return loader.load(path, [f"{key}={value}" for key, value in overrides.items()])


class FakeFetch(Stage):
    name, scope, consumes, produces = "fetch", "batch", BatchRef, RawBatch

    def output(self, ctx: Context) -> RawBatch:
        return RawBatch(ctx.layout.raw(ctx.batch))

    def run(self, source: BatchRef, ctx: Context) -> RawBatch:
        ctx.layout.raw(source).mkdir(parents=True, exist_ok=True)
        return self.output(ctx)


class FakeSegment(Stage):
    """Writes SOURCES x CLIPS_PER_SOURCE one-second clips, and records which worker and GPU ran it."""

    name, scope, consumes, produces, needs_gpu = "segment", "batch", RawBatch, ClipManifest, True
    provides = ("segment.csv",)

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "segment.csv")

    def run(self, source: RawBatch, ctx: Context) -> ClipManifest:
        if (ctx.layout.root / f"fail_{ctx.batch.index}").exists():
            raise RuntimeError("segment was told to fail")
        clips, rows = ctx.layout.clips(ctx.batch), []
        clips.mkdir(parents=True, exist_ok=True)
        for source_index in range(SOURCES):
            video = f"video{ctx.batch.index}x{source_index}"
            for clip in range(CLIPS_PER_SOURCE):
                name = f"{video}-{clip:05d}-00000000-00000100"
                sf.write(clips / f"{name}.wav", np.full(2400, clip / 10, dtype=np.float32), 2400)
                rows.append({"dataset": "YODAS2-Sidon", "name": name, "transcription": f"Clip {clip}.",
                             "speaker_id": video, "language": "en", "duration": "1.000000",
                             "audio_path": str((clips / f"{name}.wav").relative_to(ctx.layout.root)),
                             "source_id": video})
        manifest.write(ctx.work_dir / "segment.csv", rows)
        (ctx.work_dir / "ran_on.json").write_text(json.dumps(
            {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "threads": os.environ.get("OMP_NUM_THREADS")}))
        ctx.summary.update(clips_out=len(rows), hours_out=len(rows) / 3600)
        return self.output(ctx)


class FakeQuality(Stage):
    """Rejects the first clip of every source."""

    name, scope, consumes, produces, needs_gpu = "quality", "batch", ClipManifest, ClipManifest, True
    provides = ("verdicts.csv",)

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "quality.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        rows = manifest.read(source.path)
        kept = [row for row in rows if "-00000-" not in row["name"]]
        manifest.write(ctx.work_dir / "quality.csv", kept)
        ctx.summary.update(clips_in=len(rows), clips_out=len(kept), hours_out=len(kept) / 3600)
        return self.output(ctx)


def shard_job(task: tuple[str, str], advance: object) -> str:
    """A map_gpus job for tests: start a long-lived child and record its PID, then succeed, raise, wait or die."""

    behaviour, pid_file = task
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(pid_file, "a") as handle:
        handle.write(f"{behaviour} {child.pid}\n")
    deadline = time.time() + 30
    while behaviour in ("raise", "die") and time.time() < deadline:     # fail only once the other shard is up
        with open(pid_file) as handle:
            if "wait " in handle.read():
                break
        time.sleep(0.05)
    if behaviour == "raise":
        raise ValueError("this shard was told to fail")
    if behaviour == "die":
        os._exit(7)
    if behaviour == "wait":
        time.sleep(120)
    child.kill()
    return behaviour
