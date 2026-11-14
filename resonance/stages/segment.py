"""segment: RawBatch -> ClipManifest. Cuts recordings into sentence-aligned utterances with Whisper transcripts.

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
"""

import collections
import concurrent.futures
import multiprocessing
import shutil
import time

import soundfile as sf

from resonance.datasets import yodas2_sidon
from resonance.io import manifest
from resonance.pipeline import state
from resonance.pipeline.artifacts import ClipManifest, RawBatch
from resonance.pipeline.stage import Context, Stage
from resonance.segmentation import packing, sentences, transcribe, vad
from resonance.segmentation.abbreviations import ABBREVIATIONS, ORDINAL_DOT


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
        config, language = self.config.segment, self.config.source.language
        recordings = yodas2_sidon.recordings(source.path)
        if not recordings:
            raise RuntimeError(f"No recordings under {source.path}")
        clips = ctx.layout.clips(ctx.batch)
        clips.mkdir(parents=True, exist_ok=True)
        pipeline = transcribe.load(config.whisper)
        abbreviations, ordinal_dot = ABBREVIATIONS.get(language, set()), language in ORDINAL_DOT

        rows, skipped, foreign, source_hours = [], 0, 0, 0.0
        timings = dict.fromkeys(("read", "resample", "vad", "waiting", "whisper", "pack", "write"), 0.0)
        progress = ctx.log.progress(len(recordings))
        workers = self.config.resources.cpu_workers
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=vad.setup, initargs=(self.config.resources.threads_per_worker,)) as executor:
            # at most `workers` prepared recordings wait in memory: each is minutes to hours of audio
            pending = collections.deque(executor.submit(vad.prepare, (path, config.margin))
                                        for path in recordings[:workers])
            upcoming = iter(recordings[workers:])
            while pending:
                clock = time.perf_counter()
                prepared = pending.popleft().result()
                timings["waiting"] += time.perf_counter() - clock
                following = next(upcoming, None)
                if following is not None:
                    pending.append(executor.submit(vad.prepare, (following, config.margin)))
                for step, seconds in prepared["spent"].items():
                    timings[step] += seconds
                progress.advance()
                if "error" in prepared:
                    ctx.log.error(f"skipping {prepared['path'].name}: {prepared['error']}")
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
                for index, (begin, end, text) in enumerate(utterances):
                    name = (f"{meta['video_id']}-{index:05d}-{round((offset + begin) * 100):08d}-"
                            f"{round((offset + end) * 100):08d}")
                    samples = audio[round(begin * rate):round(end * rate)]
                    path = clips / f"{name}.wav"
                    sf.write(path, samples, rate)
                    rows.append({"dataset": yodas2_sidon.DATASET, "name": name, "transcription": text,
                                 "speaker_id": meta["video_id"], "language": language,
                                 "duration": f"{len(samples) / rate:.3f}",
                                 "audio_path": str(path.relative_to(ctx.layout.root)),
                                 "source_id": meta["video_id"]})
                timings["write"] += time.perf_counter() - clock
        progress.flush()

        manifest.write(ctx.work_dir / "segment.csv", rows)
        ctx.summary.update(recordings=len(recordings) - skipped, clips_out=len(rows),
                           hours_out=round(manifest.hours(rows), 3), source_hours=round(source_hours, 2),
                           foreign=foreign, timings={step: round(seconds, 1) for step, seconds in timings.items()})
        return self.output(ctx)

    def cleanup(self, source: RawBatch, ctx: Context) -> None:
        """The raw batch goes only after segment.done exists, so a crash never costs a second download."""

        if not self.config.segment.keep_raw:
            shutil.rmtree(source.path, ignore_errors=True)
            state.clear(ctx.work_dir, "fetch")
