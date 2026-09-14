"""Cut YODAS2-Sidon into utterances, filter them, and re-transcribe the survivors with Whisper.

    python scripts/filter/yodas2_sidon.py --batch train-00000 --limit 10

A YODAS2-Sidon shard unpacks to one long-form recording per video — <n>.flac beside an
<n>.metadata.json holding the caption timestamps. --batch picks one shard directory under --root,
--limit a random sample of the recordings inside it.

A caption is not an utterance: they run 2.6 s at the median, so cutting on them as shipped throws
away three quarters of the corpus on the duration floor alone. Consecutive captions are joined
first — while the gap between them stays under --max-gap and the chunk stays under
--target-duration — and it is those chunks that are cut and judged. On this corpus the gap rule
almost never fires: captions tile the video with a median gap of 0.01 s, so --target-duration is
what decides where a chunk ends.

The corpus's own caption text is never written out. It carries punctuation on 0.1% of utterances
and is cased on 1.7%, which is unusable as a TTS target, so an utterance that survives the
cascade is re-transcribed with Whisper and it is that transcript which lands in the CSV.

A chunk with no signal at all is dropped before the cascade rather than inside it: 1.4% of them
are digitally silent, and a stage that meets one can only report a failure, one printed line per
chunk. A caption boundary is not a speech boundary either, so a chunk can open or close on a long stretch of
music or dead air that every cascade stage passes happily — the stages judge the audio that is
there, not the audio that is missing. Silero VAD is run on what survives: dead ends longer than
--max-silence are cut back to --keep-silence, and a clip with a hole that long in the *middle* is
rejected outright, since nothing can be trimmed to save it.

**Whisper runs last on purpose.** It is the only stage that costs a GPU transcription, so the
DSP, NISQA and diarisation stages run first and roughly half the utterances are gone before a
transcript is ever asked for. The CTC stage then aligns the clip against what Whisper heard
rather than against the caption.

That split is also what --workers parallelises: the cheap cascade runs in a worker pool, while
Whisper and the aligner stay in the parent. One Whisper per worker would contend on CUDA and
cannot exist at all under MLX, which holds a single GPU context per process. Results come back in
order rather than as they finish, because the source pass below needs a recording's chunks
together. The pool is spawned rather than forked, since the parent holds a CUDA context by then;
each worker still builds its own NISQA and segmentation models on the GPU, so pin them to the CPU
in the config if the card is tight.

After a recording's last chunk, the whole recording is judged: if the diarisation stage rejected
more than config.source_max_flag_rate of the chunks it actually scored, every clip from that
recording is dropped, the way scripts/filter/emilia.py drops a leaky speaker. The per-clip stage
only sees one chunk at a time, so it cannot notice an interview whose speakers fall cleanly into
separate chunks; the rate over the recording can. Note the asymmetry with Emilia: a speaker there
is one voice in one recording, while a video here is one recording that may hold several, so
dropping a source discards its good voices along with its bad — hence a looser default than
Emilia's.

Unlike the parquet corpora this one writes its audio out: an utterance is a cut from a long-form
file and has no addressable existence anywhere else, so --audio-dir holds the accepted clips and
the later stages read them from there.
"""

import argparse
import collections
import csv
import dataclasses
import itertools
import json
import multiprocessing
import pathlib
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.filter.emilia import reached
from scripts.filter.libritts import report
from tools.codec.higgs import RESAMPLE
from tools.metrics.ctc_alignment_metric import CtcAlignmentMetric
from tools.metrics.pipeline import Pipeline
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "YODAS2-Sidon"
ROOT = pathlib.Path("data/YODAS2-sidon")
CONFIG = pathlib.Path("configurations/quality_filtering_yodas2_sidon.yaml")
AUDIO = pathlib.Path("data/processed/YODAS2-Sidon/audio")
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
SEPARATOR = "|"
MAX_GAP = 3.0                   # silence a chunk may span before it is closed, seconds
TARGET_DURATION = 8.0           # a chunk closes once it reaches this; see merge()
MODEL_RATE = 16000              # what Whisper and Silero both assume: mono float32 at 16 kHz
MAX_SILENCE = 3.0               # dead air this long is trimmed at an end, rejected in the middle
KEEP_SILENCE = 0.4              # silence left at a trimmed end, seconds
CUDA_MODEL = "large-v3-turbo"                            # faster-whisper's own name
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"       # the same weights, Metal build
PROGRESS_EVERY = 200
WINDOW_PER_WORKER = 8           # chunks queued per worker; see scored()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a single shard directory under --root, e.g. train-00000")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("yodas2_sidon_filtered.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO,
                        help="where the accepted utterances are written as wav")
    parser.add_argument("--lang", default="es", help="language code, for Whisper and the CSV")
    parser.add_argument("--max-gap", type=float, default=MAX_GAP,
                        help="join consecutive captions separated by at most this many seconds")
    parser.add_argument("--target-duration", type=float, default=TARGET_DURATION,
                        help="a joined chunk closes once it reaches this length")
    parser.add_argument("--max-silence", type=float, default=MAX_SILENCE,
                        help="dead air this long is trimmed at an end and rejected in the middle")
    parser.add_argument("--keep-silence", type=float, default=KEEP_SILENCE,
                        help="silence left in place at a trimmed end")
    parser.add_argument("--limit", type=int,
                        help="process N random recordings instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--whisper", help=f"model override (default: {CUDA_MODEL} on CUDA, "
                                          f"{MLX_MODEL} on an Apple GPU)")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel processes for the cheap cascade; each loads its own NISQA "
                             "and segmentation models, and on CUDA they share the GPU with "
                             "Whisper, so more is not faster")
    return parser.parse_args()


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    recordings = discover(args)
    expected = total_chunks(recordings, args.max_gap, args.target_duration, config.max_duration)

    print_test_title(f"Filtering {DATASET}: {len(recordings)} recordings, {expected} utterances")
    print_info("root", args.root)
    print_info("batch", args.batch or f"all {len({p.parent for p in recordings})}")
    print_info("chunks", f"captions joined across gaps up to {args.max_gap:.1f} s, "
                         f"closed at {args.target_duration:.1f} s -> {expected} utterances")
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("audio", args.audio_dir)
    print_info("workers", f"{args.workers} on the cascade, Whisper in the parent")

    print_section("Loading models")
    # The cheap stages run without a transcript, so the cascade is built with CTC switched off and
    # the aligner is held separately for the clips that earn one.
    pipeline = Pipeline(dataclasses.replace(config, ctc_enabled=False), verbose=args.verbose)
    print_info("cascade", ", ".join(stage.verdict.value for stage in pipeline.stages))
    ctc = CtcAlignmentMetric(device=config.ctc_device, uroman=config.ctc_uroman_enabled)
    whisper = Whisper(args.whisper, args.lang)
    print_info("whisper", f"{whisper.name} ({whisper.backend})")
    vad = Vad(args.max_silence, args.keep_silence)
    print_info("vad", f"silero: trim ends past {args.max_silence:.1f} s to "
                      f"{args.keep_silence:.1f} s, reject a gap that long mid-clip")
    print_info("ctc gate", "on" if config.ctc_enabled else "off (transcript kept, not gated)")
    print_info("source pass", f"drop a recording above {config.source_max_flag_rate:.0%} flagged "
                              f"clips (min {config.source_min_clips} scored)"
                              if config.source_rejection_enabled else "off")

    print_section("Filtering")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # CTC is not a pipeline stage here, but a clip that reached it passed everything before it
    order = [stage.verdict for stage in pipeline.stages] + [QualityVerdict.CTC_ALIGNMENT]
    verdicts = collections.Counter()
    accepted, failures, transcribed, start = [], 0, 0, time.time()
    source, dropped = None, []

    with open(args.output, "w", newline="") as handle:
        # Whisper emits no SEPARATOR and no backslash, so the writer never escapes; quotechar=None
        # leaves the quotes it does emit as written, matching the other filters' dialect.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        stream = utterances(recordings, args.max_gap, args.target_duration, config.max_duration)
        for name, video, audio, verdict, error in scored(stream, config, args):
            if source is not None and source.video != video:
                resolve(source, writer, args.audio_dir, verdicts, accepted, dropped, config)
                handle.flush()
                source = None
            if source is None:
                source = Source(video)

            # taken before the VAD and Whisper can overwrite the verdict, so it answers "did the
            # diarisation stage score this chunk", which is what the source tally below needs.
            # A chunk the cascade never saw reports SILENCE, which is deliberately not in `order`.
            cascaded = reached(verdict, order)
            text = None
            if not error and verdict is QualityVerdict.ACCEPTED:
                try:
                    audio, verdict = voiced(audio, vad, config)
                    if verdict is QualityVerdict.ACCEPTED:
                        transcribed += 1
                        text = whisper.transcribe(audio)
                        # nothing said is nothing to align: music and applause land here
                        if not text:
                            verdict = QualityVerdict.CTC_ALIGNMENT
                        elif config.ctc_enabled and not ctc.validate(
                            audio, rbound=config.ctc_max, transcript=text
                        ):
                            verdict = QualityVerdict.CTC_ALIGNMENT
                except Exception as failure:
                    error = str(failure)
            if error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}", flush=True)
                failures += 1
                continue

            verdicts[verdict] += 1
            done = sum(verdicts.values())
            if cascaded:
                source.scored += 1
                source.flagged += verdict is QualityVerdict.MULTI_SPEAKER
            if verdict is QualityVerdict.ACCEPTED:
                # the audio goes to disk now and the row waits: a long recording's chunks are
                # hundreds of megabytes to hold, its rows are nothing
                sf.write(args.audio_dir / f"{name}.wav", audio["audio"], audio["sample_rate"])
                source.rows.append(((DATASET, name, text, video, args.lang),
                                    len(audio["audio"]) / audio["sample_rate"], name))
            if done % PROGRESS_EVERY == 0:
                rate = done / (time.time() - start)
                kept = len(accepted) + len(source.rows)
                print(f"  {done}/{expected} utterances ({100 * done / expected:.0f}%), "
                      f"{rate:.1f}/s, {transcribed} transcribed, "
                      f"{100 * kept / done:.0f}% accepted", flush=True)

        if source is not None:
            resolve(source, writer, args.audio_dir, verdicts, accepted, dropped, config)

    if dropped:
        print_info("sources rejected",
                   f"{len(dropped)} recordings over {config.source_max_flag_rate:.0%} flagged, "
                   f"{sum(len(item.rows) for item in dropped)} of their clips")
    print_info("whisper calls", f"{transcribed} of {sum(verdicts.values())} utterances "
                                f"({100 * transcribed / max(sum(verdicts.values()), 1):.0f}%)")
    report(verdicts, accepted, failures, time.time() - start, args.output)


def voiced(audio, vad, config):
    """Trim the clip's dead ends, or reject it as silence; returns (audio, verdict).

    Rejected when the VAD hears no speech at all, when it hears a hole of vad.max_silence in the
    middle, or when trimming the ends leaves less than the cascade's own floor. All three are the
    same defect — a clip that is mostly not speech — so all three report SILENCE rather than
    borrowing TOO_SHORT, which reached() reads as never having been diarised.
    """

    trimmed = vad.trim(audio)
    if trimmed is None:
        return audio, QualityVerdict.SILENCE
    if len(trimmed["audio"]) / trimmed["sample_rate"] < config.min_duration:
        return trimmed, QualityVerdict.SILENCE
    return trimmed, QualityVerdict.ACCEPTED


class Vad:
    """Silero VAD, run on what the cascade passed and before Whisper is paid for.

    Loaded from the installed package rather than the hub, so it needs no network, and it runs on
    the CPU in single-digit milliseconds per clip — cheap enough to sit in the parent beside
    Whisper rather than in the worker pool, which returns verdicts and not audio.
    """

    def __init__(self, max_silence=MAX_SILENCE, keep=KEEP_SILENCE):
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.model = load_silero_vad()
        self.timestamps = get_speech_timestamps
        self.max_silence = max_silence
        self.keep = keep

    def trim(self, audio):
        """The clip with its dead ends cut back, or None when it should be rejected outright."""

        speech = self.timestamps(torch.from_numpy(to_model_rate(audio)), self.model,
                                 sampling_rate=MODEL_RATE, return_seconds=True)
        if not speech:
            return None
        if any(later["start"] - earlier["end"] >= self.max_silence
               for earlier, later in zip(speech, speech[1:])):
            return None

        rate = audio["sample_rate"]
        duration = len(audio["audio"]) / rate
        head, tail = speech[0]["start"], duration - speech[-1]["end"]
        begin = max(0.0, head - self.keep) if head >= self.max_silence else 0.0
        end = min(duration, speech[-1]["end"] + self.keep) if tail >= self.max_silence else duration
        if begin == 0.0 and end == duration:
            return audio
        return audio | {"audio": audio["audio"][round(begin * rate):round(end * rate)]}


def scored(stream, config, args):
    """Cheap-cascade verdicts for every chunk, in a worker pool unless --workers 1.

    Yields (name, video, audio, verdict, error). Whisper and the aligner are deliberately not in
    here — see the module docstring — so a worker returns only its verdict and the audio stays in
    the parent, which needs it for the transcription and for writing the clip out.

    The pool is fed a window at a time: a task carries the chunk's samples, ~770 kB at the default
    8 s, and Pool's task thread would otherwise drain the whole recording into memory. Results are
    ordered, not `imap_unordered`, so a recording's chunks stay together for the source pass.
    """

    if args.workers <= 1:
        _setup(config, args.verbose)
        for name, video, audio in stream:
            verdict, error = _score(audio)
            yield name, video, audio, verdict, error
        return

    size = WINDOW_PER_WORKER * args.workers
    # spawn, never the fork Linux defaults to. By this point the parent has initialised CUDA for
    # Whisper, and a forked child inherits a context it cannot use: every GPU metric in the worker
    # then raises "CUDA driver initialization failed", and MultiSpeakerMetric answers a raise with
    # a rejection — so the run would quietly reject every clip rather than fail. macOS already
    # spawns by default, which is why this only bites on a CUDA box.
    with multiprocessing.get_context("spawn").Pool(
        args.workers, _setup, (config, args.verbose)
    ) as pool:
        while window := list(itertools.islice(stream, size)):
            outcomes = pool.imap(_score, [audio for _, _, audio in window], chunksize=1)
            for (name, video, audio), (verdict, error) in zip(window, outcomes):
                yield name, video, audio, verdict, error


_pipeline = None


def _setup(config, verbose):
    """One cascade per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline
    torch.set_num_threads(1)
    _pipeline = Pipeline(dataclasses.replace(config, ctc_enabled=False), verbose=verbose)


def _score(audio):
    """One chunk through the cheap cascade; only the verdict travels back.

    A flat chunk is answered here instead of being handed to the cascade. 1.4% of chunks are
    digitally silent — caption spans do not always cover audio — and every stage that meets one
    can only call it a failure, printing a line per chunk; over a full shard that is thousands of
    them. Peak-to-peak zero is exactly the case the bandwidth metric cannot score, since its
    loud-frame filter keeps nothing when every frame carries the same energy.
    """

    if np.ptp(audio["audio"]) == 0:
        return QualityVerdict.SILENCE, None
    try:
        return _pipeline.run(audio), None
    except Exception as error:
        return None, str(error)


@dataclasses.dataclass
class Source:
    """One recording's tally, held until its last chunk has been judged."""

    video: str
    rows: list = dataclasses.field(default_factory=list)   # (csv row, duration, clip name)
    scored: int = 0                 # chunks the diarisation stage actually scored
    flagged: int = 0                # of those, the ones it rejected


def resolve(source, writer, audio_dir, verdicts, accepted, dropped, config):
    """Commit a finished recording's clips, or drop the source whole.

    The rate is taken over the chunks that reached the diarisation stage, not over every chunk:
    one rejected on length was never diarised and says nothing about the recording. Clips of a
    dropped source are unlinked, since their audio was written as it was accepted.
    """

    if not flagged_source(source, config):
        for row, duration, _ in source.rows:
            writer.writerow(row)
            accepted.append((source.video, duration))
        return

    for _, _, name in source.rows:
        (audio_dir / f"{name}.wav").unlink(missing_ok=True)
    verdicts[QualityVerdict.ACCEPTED] -= len(source.rows)
    verdicts[QualityVerdict.MULTI_SPEAKER_SOURCE] += len(source.rows)
    dropped.append(source)


def flagged_source(source, config):
    """Whether the diarisation stage rejected enough of this recording to distrust all of it."""

    return (config.source_rejection_enabled
            and source.scored >= config.source_min_clips
            and source.flagged / source.scored >= config.source_max_flag_rate)


def discover(args):
    """The recordings to cut, as metadata paths: one shard directory, or every one under --root."""

    directory = args.root / args.batch if args.batch else args.root
    if not directory.is_dir():
        raise SystemExit(f"No shard directory {directory}")
    found = sorted(directory.rglob("*.metadata.json"))
    if not found:
        raise SystemExit(f"No {DATASET} recordings under {directory}")
    if args.limit and args.limit < len(found):
        found = sorted(random.Random(args.seed).sample(found, args.limit))
    return found


def utterances(recordings, max_gap, target, limit):
    """Every joined chunk of every recording, cut from its flac.

    The file is opened once per recording and seeked, rather than reopened per chunk: a recording
    runs to well over an hour and its flac header is not free to parse.
    """

    for meta_path in recordings:
        meta = json.loads(meta_path.read_text())
        flac = meta_path.with_name(meta_path.name.replace(".metadata.json", ".flac"))
        if not flac.exists():
            print(f"{Colors.WARNING}  no audio for {meta_path.name}{Colors.ENDC}")
            continue
        with sf.SoundFile(flac) as handle:
            rate = handle.samplerate
            for name, begin, end in chunk_spans(meta, max_gap, target, limit,
                                                handle.frames / rate):
                handle.seek(int(begin * rate))
                y = handle.read(int((end - begin) * rate), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                yield name, meta["video_id"], {"audio": y, "sample_rate": rate}


def chunk_spans(meta, max_gap, target, limit, duration=None):
    """One recording's joined chunks, as (name, start, end).

    Caption timestamps are not bounded by the audio: on the Spanish batch the spans cover 101% of
    the stated duration at the 90th percentile. A chunk starting at or past the end is dropped and
    one merely ending past it is clamped, because SoundFile.seek() answers a position beyond the
    file with `Internal psf_fseek() failed` rather than with a short read. Pass duration=None to
    count the spans as the metadata states them.
    """

    spans = meta.get("utterances") or {}
    chunks = merge(list(zip(spans.get("utt_id", []), spans.get("start", []), spans.get("end", []))),
                   max_gap, target, limit)
    if duration is None:
        return chunks
    return [(name, begin, min(end, duration)) for name, begin, end in chunks if begin < duration]


def total_chunks(recordings, max_gap, target, limit):
    """How many chunks the run will judge, walked from the metadata without opening any audio.

    The progress line needs a denominator, and the caption count is not it: joining collapses
    roughly three captions into one chunk. Recordings whose flac is missing are skipped here as
    utterances() skips them, and each audio header is read so that chunks past the end of their
    recording are dropped from the count exactly as they are from the run — so the total is what
    will actually be reached.
    """

    total = 0
    for meta_path in recordings:
        flac = meta_path.with_name(meta_path.name.replace(".metadata.json", ".flac"))
        try:
            duration = sf.info(flac).duration
        except Exception:                       # missing or unreadable; utterances() warns on it
            continue
        total += len(chunk_spans(json.loads(meta_path.read_text()), max_gap, target, limit,
                                 duration))
    return total


def merge(spans, max_gap, target, limit):
    """Join consecutive captions into chunks, as (name, start, end).

    A chunk grows until it reaches `target`, and is cut short of that by a gap wider than
    `max_gap` or by `limit`, the cascade's own ceiling — a chunk past that is rejected on length
    whatever else is true of it. The chunk keeps the first caption's id with its end timestamp
    rewritten, which is the corpus's own naming scheme and stays unique.
    """

    out, name, begin, end = [], None, 0.0, 0.0
    for caption, start, stop in spans:
        if name is not None and (start - end > max_gap or stop - begin > limit):
            out.append((rename(name, end), begin, end))
            name = None
        if name is None:
            name, begin = caption, start
        end = stop
        if end - begin >= target:
            out.append((rename(name, end), begin, end))
            name = None
    if name is not None:
        out.append((rename(name, end), begin, end))
    return out


def rename(caption, end):
    """'<video>-00000-00001131-00001320' with the end field moved to where the chunk now ends."""

    return f"{caption.rsplit('-', 1)[0]}-{round(end * 100):08d}"


class Whisper:
    """Punctuated transcripts, on whichever accelerator this machine has.

    CUDA gets faster-whisper, which is the fast path on that hardware; an Apple GPU gets
    mlx-whisper, the only one of the two with a Metal backend. Both are imported lazily, so
    neither has to be installed on a machine that will not use it.
    """

    def __init__(self, model=None, language="es"):
        self.language = language
        self.mlx = None
        if torch.cuda.is_available():
            from faster_whisper import WhisperModel

            self.backend, self.name = "cuda", model or CUDA_MODEL
            self.model = WhisperModel(self.name, device="cuda", compute_type="float16")
        elif torch.backends.mps.is_available():
            import mlx_whisper

            self.backend, self.name = "mps", model or MLX_MODEL
            self.mlx = mlx_whisper
        else:
            from faster_whisper import WhisperModel

            self.backend, self.name = "cpu", model or CUDA_MODEL
            self.model = WhisperModel(self.name, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        """What the clip actually says, punctuated and cased; empty when Whisper hears no speech."""

        y = to_model_rate(audio)
        if self.mlx is not None:
            spoken = self.mlx.transcribe(y, path_or_hf_repo=self.name, language=self.language,
                                         verbose=None)["text"]
        else:
            segments, _ = self.model.transcribe(y, language=self.language)
            spoken = " ".join(segment.text for segment in segments)
        return " ".join(spoken.split())


def to_model_rate(audio):
    """Whisper and Silero both want 16 kHz; the corpus is 24 kHz. The codec's kaiser parameters."""

    y = np.asarray(audio["audio"], dtype=np.float32)
    if audio["sample_rate"] == MODEL_RATE:
        return y
    return torchaudio.functional.resample(torch.from_numpy(y), audio["sample_rate"],
                                          MODEL_RATE, **RESAMPLE).numpy()


if __name__ == "__main__":
    main()
