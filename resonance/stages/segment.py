"""
segment: RawBatch -> ClipManifest. Cuts recordings into sentence-aligned utterances with Whisper transcripts.

A recording is one long-form video. Its captions are not utterances: they start and end mid-sentence, and their
timestamps cut into speech. So the captions only say where speech is, and the recording is re-segmented from
Whisper's own words:

1. Only the captioned span, widened by `margin`, is read.
2. Silero VAD finds the speech in it; its segments are grouped into 30 s windows that Whisper transcribes in
   batches with word timestamps.
3. The words are split into sentences, and whole sentences are packed into utterances of about `target_duration`
   seconds, cut only in real pauses (resonance/segmentation/packing.py).
4. An utterance overlapping a window whose detected language is not the corpus language is thrown away.

Each utterance is written as a wav at the recording's native rate under clips/, and listed in segment.csv.

`segment.workers` Whisper loops share the GPU, each in its own process with its own model and its own share of
the recordings. A recording's result does not depend on which loop takes it, so the manifest is the same for
any number of loops.
"""

import collections
import concurrent.futures
import dataclasses
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import soundfile as sf

from resonance.config.schema import SegmentConfig
from resonance.datasets import yodas2_sidon
from resonance.io import manifest
from resonance.pipeline import gpu, state
from resonance.pipeline.artifacts import ClipManifest, RawBatch
from resonance.pipeline.stage import Context, Stage
from resonance.segmentation import packing, sentences, transcribe, vad
from resonance.segmentation.abbreviations import ABBREVIATIONS, ORDINAL_DOT


@dataclasses.dataclass
class Share:
    """What one Whisper loop is given: its recordings, with their position in the batch, and where to write."""

    recordings: list[tuple[int, Path]]
    config: SegmentConfig
    language: str
    clips: Path
    root: Path
    prep_workers: int
    threads: int


def segment_recordings(share: Share, advance: Callable[[int], None]) -> dict:
    """
    One Whisper loop: prepare, transcribe, pack and write every recording of the share.

    Returns the manifest rows per recording position, the counters, the time spent per step, and the recordings
    that could not be read. Runs in its own process, so it reports through `advance` and its return value only.
    """

    config, language = share.config, share.language
    pipeline = transcribe.load(config.whisper)
    abbreviations, ordinal_dot = ABBREVIATIONS.get(language, set()), language in ORDINAL_DOT
    rows, unreadable, skipped, foreign, source_hours = {}, [], 0, 0, 0.0
    timings = dict.fromkeys(("read", "resample", "vad", "waiting", "whisper", "pack", "write"), 0.0)
    workers = share.prep_workers
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=vad.setup, initargs=(share.threads,)) as executor:
        # at most `workers` prepared recordings wait in memory: each is minutes to hours of audio
        pending = collections.deque((position, executor.submit(vad.prepare, (path, config.margin)))
                                    for position, path in share.recordings[:workers])
        upcoming = iter(share.recordings[workers:])
        while pending:
            clock = time.perf_counter()
            position, future = pending.popleft()
            prepared = future.result()
            timings["waiting"] += time.perf_counter() - clock
            following = next(upcoming, None)
            if following is not None:
                pending.append((following[0], executor.submit(vad.prepare, (following[1], config.margin))))
            for step, seconds in prepared["spent"].items():
                timings[step] += seconds
            advance(1)
            if "error" in prepared:
                unreadable.append(f"skipping {prepared['path'].name}: {prepared['error']}")
            if prepared.get("loaded") is None:
                skipped += 1
                continue
            meta, audio, rate, offset, duration = prepared["loaded"]
            source_hours += duration / 3600
            if not prepared["speech"]:
                continue

            clock = time.perf_counter()
            words, foreign_windows = transcribe.transcribe(pipeline, prepared["y"], prepared["speech"],
                                                           language, config.batch_size, config.beam_size)
            timings["whisper"] += time.perf_counter() - clock

            clock = time.perf_counter()
            found = sentences.split(words, config.max_pause, abbreviations, ordinal_dot)
            silences = packing.gaps(prepared["speech"], len(prepared["y"]) / vad.MODEL_RATE, config.min_gap)
            utterances = []
            for begin, end, text in packing.pack(found, silences, config):
                if transcribe.foreign_overlap(begin, end, foreign_windows):
                    foreign += 1
                else:
                    utterances.append((begin, end, text))
            timings["pack"] += time.perf_counter() - clock

            clock = time.perf_counter()
            rows[position] = []
            for index, (begin, end, text) in enumerate(utterances):
                name = (f"{meta['video_id']}-{index:05d}-{round((offset + begin) * 100):08d}-"
                        f"{round((offset + end) * 100):08d}")
                samples = audio[round(begin * rate):round(end * rate)]
                path = share.clips / f"{name}.wav"
                sf.write(path, samples, rate)
                rows[position].append({"dataset": yodas2_sidon.DATASET, "name": name, "transcription": text,
                                       "speaker_id": meta["video_id"], "language": language,
                                       "duration": f"{len(samples) / rate:.6f}",
                                       "audio_path": str(path.relative_to(share.root)),
                                       "source_id": meta["video_id"]})
            timings["write"] += time.perf_counter() - clock
    return {"rows": rows, "unreadable": unreadable, "skipped": skipped, "foreign": foreign,
            "source_hours": source_hours, "timings": timings}


class Segment(Stage):
    name = "segment"
    scope = "batch"
    consumes = RawBatch
    produces = ClipManifest
    needs_gpu = True
    provides = ("segment.csv",)

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "segment.csv")

    def run(self, source: RawBatch, ctx: Context) -> ClipManifest:
        config, resources = self.config.segment, self.config.resources
        recordings = yodas2_sidon.recordings(source.path)
        if not recordings:
            raise RuntimeError(f"No recordings under {source.path}")
        clips = ctx.layout.clips(ctx.batch)
        clips.mkdir(parents=True, exist_ok=True)

        parts = gpu.split(list(enumerate(recordings)), config.workers)
        prep = max(1, min(resources.cpu_workers, os.cpu_count() or 1) // len(parts))
        shares = [Share(part, config, self.config.source.language, clips, ctx.layout.root, prep,
                        resources.threads_per_worker) for part in parts]
        # every loop inherits this process's GPU: the scheduler has already pinned it
        results = gpu.map_gpus(segment_recordings, shares, [None], resources.threads_per_worker,
                               ctx.log.progress(len(recordings)))

        for message in (message for result in results for message in result["unreadable"]):
            ctx.log.error(message)
        by_position = {position: rows for result in results for position, rows in result["rows"].items()}
        rows = [row for position in sorted(by_position) for row in by_position[position]]
        timings = {step: round(sum(result["timings"][step] for result in results), 1)
                   for step in results[0]["timings"]}
        manifest.write(ctx.work_dir / "segment.csv", rows)
        ctx.summary.update(recordings=len(recordings) - sum(result["skipped"] for result in results),
                           clips_out=len(rows), hours_out=round(manifest.hours(rows), 3),
                           source_hours=round(sum(result["source_hours"] for result in results), 2),
                           foreign=sum(result["foreign"] for result in results), whisper_loops=len(parts),
                           timings=timings)
        return self.output(ctx)

    def cleanup(self, source: RawBatch, ctx: Context) -> None:
        """The raw batch goes only after segment.done exists, so a crash never costs a second download."""

        if not self.config.segment.keep_raw:
            shutil.rmtree(source.path, ignore_errors=True)
            state.clear(ctx.work_dir, "fetch")
