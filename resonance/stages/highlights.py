"""
highlights: ClipManifest -> ClipManifest. Picks the most expressive, most aroused and most unusual-voiced clips
into a small second dataset beside the main one.

    runs/<run>/highlights/<run>.csv     list, rank, the standard columns and the scores of every picked clip
    runs/<run>/highlights/audio.zip     <list>/<rank>_<name>.wav, copied
    runs/<run>/highlights/scores.csv    the scores of every clip, so lists can be re-cut without scoring again

Every clip is scored for prosody (pitch and loudness spread, on the CPU), arousal (a wav2vec2 emotion model) and
speaker embedding (WavLM). The manifest passes through unchanged.
"""

import multiprocessing
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from resonance.highlights import models, prosody, ranking
from resonance.io import manifest
from resonance.pipeline import gpu
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage

LISTS = ("expressive_top", "arousal_top", "voice_unique_top")
SCORES = ("expressive", "arousal", "voice_unique", *prosody.FIELDS)
PROSODY_WORKERS = 64


class Highlights(Stage):
    name = "highlights"
    scope = "pool"
    consumes = ClipManifest
    produces = ClipManifest
    needs_gpu = True

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "highlights.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config, threads = self.config.highlights, self.config.resources.threads_per_worker
        rows = manifest.read(source.path)
        paths = [str(ctx.layout.root / row["audio_path"]) for row in rows]
        speakers = [row["speaker_id"] for row in rows]

        tasks = [(path, len(row["transcription"].split())) for path, row in zip(paths, rows)]
        # by length, so the clips sharing a batch need little padding, and longest first: growing batches would
        # each need new GPU memory, which the loops sharing a GPU run out of. Dealt out, every loop gets the same mix
        order = sorted(range(len(rows)), key=lambda index: float(rows[index]["duration"]), reverse=True)
        shares = gpu.split(order, len(ctx.gpus) * config.workers)

        # prosody on the CPU cores while the GPUs score arousal and voice
        with gpu.pinned(None, 1):
            pool = ProcessPoolExecutor(min(PROSODY_WORKERS, os.cpu_count() or 1),
                                       multiprocessing.get_context("spawn"))
            try:
                # one clip alone first: workers compiling librosa's numba code into a cold cache together crash
                measured = [pool.submit(prosody.features, tasks[0]).result()]
                pending = pool.map(prosody.features, tasks[1:], chunksize=32)
                parts = gpu.map_gpus(models.score_clips, [[paths[index] for index in share] for share in shares],
                                     ctx.gpus, threads, ctx.log.progress(len(rows)), self.log_path(ctx))
                measured += pending
            finally:
                pool.shutdown(cancel_futures=True)

        scores = {field: np.array([row[field] for row in measured]) for field in prosody.FIELDS}
        scores["arousal"] = np.empty(len(rows))
        embeddings = np.empty((len(rows), parts[0][1].shape[1]), dtype=np.float32)
        for share, (arousal, vectors) in zip(shares, parts):
            scores["arousal"][share], embeddings[share] = arousal, vectors
        scores["expressive"] = ranking.expressive(scores)
        scores["voice_unique"], typical = ranking.voice_unique(embeddings, speakers)
        lists = {"expressive_top": ranking.top(scores["expressive"], speakers, config.per_list, config.per_speaker),
                 "arousal_top": ranking.top(scores["arousal"], speakers, config.per_list, config.per_speaker),
                 "voice_unique_top": ranking.unique_voices(scores["voice_unique"], typical, speakers,
                                                           config.per_list)}

        folder = ctx.layout.highlights
        folder.mkdir(parents=True, exist_ok=True)
        rounded = [{field: f"{scores[field][index]:.4f}" for field in SCORES} for index in range(len(rows))]
        manifest.write(folder / "scores.csv", [{"name": row["name"], **values} for row, values in zip(rows, rounded)],
                       ("name", *SCORES))
        picked = [{**rows[index], **rounded[index], "list": name, "rank": str(rank)}
                  for name in LISTS for rank, index in enumerate(lists[name], 1)]
        manifest.write(folder / f"{self.config.run.name}.csv", picked,
                       ["list", "rank", *manifest.columns_of(rows, manifest.STANDARD), *SCORES])
        with zipfile.ZipFile(folder / "audio.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in LISTS:
                for rank, index in enumerate(lists[name], 1):
                    archive.write(paths[index], f"{name}/{rank:02d}_{rows[index]['name']}.wav")

        manifest.write(ctx.work_dir / "highlights.csv", rows)
        ctx.summary.update(clips_out=len(rows), hours_out=round(manifest.hours(rows), 3), scored=len(rows),
                           picked=len(picked), speakers=len(set(speakers)))
        return self.output(ctx)
