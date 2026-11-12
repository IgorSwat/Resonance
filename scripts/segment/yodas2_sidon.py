"""Segment YODAS2-Sidon recordings into sentence-aligned utterances with Whisper transcripts.

    python scripts/segment/yodas2_sidon.py --root data/YODAS2-sidon/en --batch 000/train-00000 --lang en

A recording is one long-form video (<n>.flac beside <n>.metadata.json). Its captions are not
utterances: they start and end mid-sentence, and their timestamps cut into speech. So the captions
are used only to find where speech is, and the recording is re-segmented from Whisper's own words:

1. Only the span between the first caption's start and the last caption's end, widened by
   --margin, is read. The rest of the video holds nothing the corpus labelled as speech.
2. Silero VAD (the ONNX build bundled with faster-whisper) finds the speech in that span. Its
   segments are grouped into windows of at most 30 s, which Whisper large-v3-turbo transcribes in
   batches with word timestamps.
3. The words are split into sentences on . ? !, and consecutive whole sentences are packed into
   utterances of about --target-duration seconds, never across a pause longer than --max-pause and
   never past --max-duration.
4. An utterance may only start or end at a real pause: a VAD silence of at least --min-gap at the
   sentence boundary. There it is cut with --lead seconds of silence before the speech and --trail
   seconds after, as far as the pause allows.
5. Whisper detects the language of every window, from the encoder output it already computed. An
   utterance overlapping a window whose top language is not --lang is thrown away: decoding is
   forced to --lang, so Whisper writes foreign speech as a translation that no later stage can
   tell from a real transcript.

A sentence Whisper did not close with punctuation, or one interrupted by a long pause, is never
part of an utterance: it is where speech was cut, overlapped or abandoned. Each utterance is written
as a wav at the recording's native rate, and its transcript to --output, in the column layout of the
filter scripts.
"""

import argparse
import collections
import concurrent.futures
import csv
import dataclasses
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
from tools.codec.higgs import RESAMPLE

DATASET = "YODAS2-Sidon"
ROOT = pathlib.Path("data/YODAS2-sidon")
AUDIO = pathlib.Path("data/processed/YODAS2-Sidon/segments")
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
SEPARATOR = "|"
MARGIN = 2.0                    # read this much beyond the first and last caption, seconds
WINDOW = 30.0                   # Whisper's input length; VAD segments are grouped up to it
TARGET_DURATION = 8.0           # an utterance closes at the first clean boundary past this
MIN_DURATION = 3.0
MAX_DURATION = 25.0
MAX_PAUSE = 2.0                 # a longer pause inside an utterance ends it
MIN_GAP = 0.15                  # a cut needs at least this much silence, seconds
TOLERANCE = 0.3                 # how far a word timestamp may sit from the VAD boundary it matches
LEAD = 0.1                      # silence kept before the speech; OmniVoice's prompt keeps 100 ms
TRAIL = 0.2                     # and after it; the prompt keeps 200 ms
MODEL_RATE = 16000              # what Whisper and Silero both assume: mono float32 at 16 kHz
CUDA_MODEL = "large-v3-turbo"   # faster-whisper's own name
BATCH_SIZE = 16
WORKERS = 8                    # processes reading and VAD-ing recordings ahead of Whisper
WORKER_THREADS = 4              # torch threads per worker, for the resampling
TERMINAL = (".", "?", "!")
CLOSING = "\"'”’)]"             # trailing characters that may follow a sentence's final mark
OPENING = "¿¡«\"“'(["           # leading characters that may precede a sentence's first letter
# Words ending in a period that do not end a sentence, per --lang. Kept apart rather than merged:
# Polish "ok." would otherwise stop "That's ok." from ending an English sentence.
ABBREVIATIONS = {
    "en": {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "prof.", "jr.", "sr."},
    "pl": {"np.", "itd.", "itp.", "tzn.", "tj.", "tzw.", "m.in.", "ul.", "al.", "dr.", "prof.",
           "mgr.", "inż.", "hab.", "płk.", "gen.", "ks.", "św.", "nr.", "pkt.", "str.", "godz.", "tys.",
           "mln.", "mld.", "zł.", "wg.", "ds.", "cdn.", "por.", "zob.", "tel.", "jw."},
    "es": {"sr.", "sra.", "srta.", "sres.", "dr.", "dra.", "ud.", "uds.", "vd.", "vds.", "d.", "dña.",
           "pág.", "págs.", "núm.", "art.", "cap.", "etc.", "ej.", "p.ej.", "aprox.", "lic.", "ing.",
           "prof.", "av.", "avda.", "fig.", "sto.", "sta.", "dto.", "tel.", "vol."},
    "pt": {"sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "profa.", "eng.", "av.", "pág.", "págs.", "núm.",
           "art.", "cap.", "etc.", "ex.", "aprox.", "fig.", "tel.", "vol.", "sto.", "sta.", "lda.", "cia."},
    "de": {"z.b.", "bzw.", "usw.", "d.h.", "u.a.", "ca.", "dr.", "prof.", "nr.", "str.", "vgl.", "evtl.", "ggf.",
           "inkl.", "zzgl.", "bspw.", "etc.", "hr.", "fr.", "jh.", "mio.", "mrd.", "tsd.", "min.", "std.", "abs.",
           "o.ä.", "u.ä.", "z.t.", "s.", "bzgl.", "allg.", "max.", "mind."},
    "it": {"sig.", "sig.ra", "sigg.", "dott.", "dott.ssa", "prof.", "prof.ssa", "ing.", "avv.", "arch.", "geom.",
           "ecc.", "es.", "pag.", "pagg.", "art.", "n.", "nr.", "cap.", "vol.", "tel.", "ca.", "p.es.", "c.a.",
           "s.p.a.", "s.r.l.", "fig.", "cfr.", "rag.", "on.", "mons.", "gen.", "col.", "magg."},
    "fr": {"m.", "mm.", "mme.", "mlle.", "dr.", "pr.", "me.", "mgr.", "st.", "ste.", "etc.", "cf.", "p.ex.", "ex.",
           "env.", "chap.", "art.", "fig.", "vol.", "p.", "pp.", "av.", "bd.", "tél.", "n°.", "c.-à-d.", "c.à.d.",
           "apr.", "av.-j.-c.", "j.-c.", "max.", "min.", "approx."},
}
# Languages whose ordinals are a number with a dot (am 3. Oktober): such a word never ends a sentence.
ORDINAL_DOT = {"de"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a shard directory under --root, e.g. 000/train-00000")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("yodas2_sidon_segments.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO,
                        help="where the utterances are written as wav")
    parser.add_argument("--lang", default="en", help="language code, for Whisper and the CSV")
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--target-duration", type=float, default=TARGET_DURATION)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION)
    parser.add_argument("--max-pause", type=float, default=MAX_PAUSE)
    parser.add_argument("--min-gap", type=float, default=MIN_GAP)
    parser.add_argument("--lead", type=float, default=LEAD)
    parser.add_argument("--trail", type=float, default=TRAIL)
    parser.add_argument("--whisper", default=CUDA_MODEL, help="faster-whisper model name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Whisper windows per batch")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="processes that read, resample and VAD recordings ahead of Whisper")
    parser.add_argument("--limit", type=int, help="process N random recordings instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    return parser.parse_args()


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


def to_model_rate(audio):
    """Whisper and Silero both want 16 kHz; the corpus is 24 kHz. The codec's kaiser parameters."""

    y = np.asarray(audio["audio"], dtype=np.float32)
    if audio["sample_rate"] == MODEL_RATE:
        return y
    return torchaudio.functional.resample(torch.from_numpy(y), audio["sample_rate"],
                                          MODEL_RATE, **RESAMPLE).numpy()


@dataclasses.dataclass
class Word:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class Sentence:
    words: list
    closed: bool                # ends on a terminal mark rather than on a long pause or the span end

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def main() -> None:
    args = parse_args()
    from faster_whisper import WhisperModel

    recordings = discover(args)
    print_test_title(f"Segmenting {DATASET}: {len(recordings)} recordings")
    print_info("whisper", f"{args.whisper}, batch {args.batch_size}, beam {args.beam_size}")
    print_info("utterances", f"target {args.target_duration} s, {args.min_duration}-"
                             f"{args.max_duration} s, pauses under {args.max_pause} s")
    print_info("edges", f"cut in a pause of at least {args.min_gap} s, keeping {args.lead} s before "
                        f"and {args.trail} s after")
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    pipeline = make_pipeline(model)

    timings = dict.fromkeys(("read", "resample", "vad", "waiting", "whisper", "segment", "write"), 0.0)
    stats = {"recordings": 0, "source_hours": 0.0, "read_hours": 0.0, "utterances": 0,
             "utterance_hours": 0.0, "sentences": 0, "open_sentences": 0, "skipped": 0,
             "foreign": 0, "foreign_hours": 0.0}
    began = time.perf_counter()

    with args.output.open("w", newline="") as handle, \
            concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=torch.set_num_threads, initargs=(WORKER_THREADS,)) as executor:
        writer = csv.writer(handle, delimiter=SEPARATOR, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        pending = collections.deque(executor.submit(prepare, path, args.margin) for path in recordings[:args.workers])
        upcoming = iter(recordings[args.workers:])
        while pending:
            clock = time.perf_counter()
            meta_path, loaded, y, speech, spent = pending.popleft().result()
            timings["waiting"] += time.perf_counter() - clock
            following = next(upcoming, None)
            if following is not None:
                pending.append(executor.submit(prepare, following, args.margin))
            for stage, seconds in spent.items():
                timings[stage] += seconds
            if loaded is None:
                stats["skipped"] += 1
                continue
            meta, audio, rate, offset, duration = loaded
            stats["recordings"] += 1
            stats["source_hours"] += duration / 3600
            stats["read_hours"] += len(audio) / rate / 3600
            if not speech:
                continue

            clock = time.perf_counter()
            words, foreign = transcribe(pipeline, y, speech, args)
            timings["whisper"] += time.perf_counter() - clock

            clock = time.perf_counter()
            sentences = split(words, args.max_pause, ABBREVIATIONS.get(args.lang, set()),
                              args.lang in ORDINAL_DOT)
            silences = gaps(speech, len(y) / MODEL_RATE, args.min_gap)
            utterances = []
            # an utterance touching a window Whisper hears in another language is thrown away
            for begin, end, text in pack(sentences, silences, args):
                if foreign_overlap(begin, end, foreign):
                    stats["foreign"] += 1
                    stats["foreign_hours"] += (end - begin) / 3600
                else:
                    utterances.append((begin, end, text))
            stats["sentences"] += len(sentences)
            stats["open_sentences"] += sum(not s.closed for s in sentences)
            timings["segment"] += time.perf_counter() - clock

            clock = time.perf_counter()
            for index, (begin, end, text) in enumerate(utterances):
                name = (f"{meta['video_id']}-{index:05d}-{round((offset + begin) * 100):08d}-"
                        f"{round((offset + end) * 100):08d}")
                sf.write(args.audio_dir / f"{name}.wav",
                         audio[round(begin * rate):round(end * rate)], rate)
                writer.writerow((DATASET, name, text, meta["video_id"], args.lang))
                stats["utterances"] += 1
                stats["utterance_hours"] += (end - begin) / 3600
            timings["write"] += time.perf_counter() - clock
            print(f"  {stats['recordings']}/{len(recordings)} {meta_path.name}: "
                  f"{len(utterances)} utterances, {stats['utterance_hours']:.2f} h so far, "
                  f"{time.perf_counter() - began:.0f} s", flush=True)

    report(stats, timings, time.perf_counter() - began, args)


def prepare(meta_path: pathlib.Path, margin: float):
    """Read, resample and VAD one recording, in a worker process beside Whisper.

    A process rather than a thread: faster-whisper's VAD steps through the audio in Python and holds
    the GIL, so threads queue behind each other. The pool is spawned because the parent holds CUDA.
    """

    from faster_whisper.vad import VadOptions, get_speech_timestamps

    clock = time.perf_counter()
    loaded = read(meta_path, margin)
    spent = {"read": time.perf_counter() - clock}
    if loaded is None:
        return meta_path, None, None, None, spent
    meta, audio, rate, offset, duration = loaded
    clock = time.perf_counter()
    y = to_model_rate({"audio": audio, "sample_rate": rate})
    spent["resample"] = time.perf_counter() - clock
    clock = time.perf_counter()
    options = VadOptions(min_silence_duration_ms=100, speech_pad_ms=0)
    speech = [(s["start"] / MODEL_RATE, s["end"] / MODEL_RATE) for s in get_speech_timestamps(y, options)]
    spent["vad"] = time.perf_counter() - clock
    return meta_path, loaded, y, speech, spent


def read(meta_path: pathlib.Path, margin: float):
    """The recording's captioned span widened by margin, as (meta, audio, rate, offset, duration)."""

    flac = meta_path.with_name(meta_path.name.replace(".metadata.json", ".flac"))
    try:
        meta = json.loads(meta_path.read_text())
        spans = meta.get("utterances") or {}
        if not spans.get("start"):
            return None
        with sf.SoundFile(flac) as handle:
            rate, duration = handle.samplerate, handle.frames / handle.samplerate
            begin = max(0.0, min(spans["start"]) - margin)
            end = min(duration, max(spans["end"]) + margin)
            if end <= begin:
                return None
            handle.seek(int(begin * rate))
            audio = handle.read(int((end - begin) * rate), dtype="float32")
    except Exception as error:
        print(f"{Colors.WARNING}  skipping {meta_path.name}: {error}{Colors.ENDC}", flush=True)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return meta, audio, rate, begin, duration


def windows(speech: list) -> list:
    """VAD segments grouped into Whisper windows of at most WINDOW seconds, in seconds."""

    grouped = []
    for start, end in speech:
        while end - start > WINDOW:     # one segment longer than a window is split at WINDOW
            grouped.append({"start": start, "end": start + WINDOW})
            start += WINDOW
        if grouped and end - grouped[-1]["start"] <= WINDOW:
            grouped[-1]["end"] = end
        else:
            grouped.append({"start": start, "end": end})
    return grouped


def transcribe(pipeline, y: np.ndarray, speech: list, args: argparse.Namespace):
    """Every word Whisper hears in the span, with its timestamps in span seconds, and the windows
    whose language Whisper detects as something other than --lang, as (start, end) in span seconds."""

    grouped = windows(speech)
    pipeline.languages = []
    segments, _ = pipeline.transcribe(y, language=args.lang, batch_size=args.batch_size,
                                      beam_size=args.beam_size, word_timestamps=True,
                                      clip_timestamps=grouped)
    # kept as Whisper spells it, leading space included: a word fragment such as "-term" or ".S."
    # comes without one, and joining stripped tokens with spaces wrote "long -term" and "U .S."
    words = [Word(w.start, w.end, w.word)
             for segment in segments for w in (segment.words or []) if w.word.strip()]
    if len(pipeline.languages) != len(grouped):
        raise RuntimeError(f"{len(pipeline.languages)} language detections for {len(grouped)} windows")
    foreign = [(window["start"], window["end"]) for window, language in zip(grouped, pipeline.languages)
               if language != args.lang]
    return words, foreign


def make_pipeline(model):
    """faster-whisper's batched pipeline, also recording the language Whisper detects in each window.

    Decoding stays forced to --lang, so a foreign window comes out as an English *translation*
    rather than as its own language — nothing in the text gives it away. The language is read off
    the encoder output the batch has already computed, which costs one decoder step per window.
    """

    from faster_whisper import BatchedInferencePipeline

    class Pipeline(BatchedInferencePipeline):
        def generate_segment_batched(self, features, tokenizer, options):
            encoder_output, output = super().generate_segment_batched(features, tokenizer, options)
            self.languages.extend(probabilities[0][0].strip("<|>")
                                  for probabilities in self.model.model.detect_language(encoder_output))
            return encoder_output, output

    return Pipeline(model)


def foreign_overlap(begin: float, end: float, foreign: list) -> bool:
    return any(start < end and begin < stop for start, stop in foreign)


def split(words: list, max_pause: float, abbreviations: set, ordinal_dot: bool = False) -> list:
    """Sentences, closed on a terminal mark; a pause over max_pause leaves the sentence open."""

    sentences, current = [], []
    for word in words:
        if current and word.start - current[-1].end > max_pause:
            sentences.append(Sentence(current, closed=False))
            current = []
        current.append(word)
        if terminal(word.text, abbreviations, ordinal_dot):
            sentences.append(Sentence(current, closed=True))
            current = []
    if current:
        sentences.append(Sentence(current, closed=False))
    return sentences


def terminal(text: str, abbreviations: set, ordinal_dot: bool = False) -> bool:
    """Whether a word ends a sentence: a final . ? ! that is not a known abbreviation (nor, with
    ordinal_dot, a number with a dot)."""

    stripped = text.strip().rstrip(CLOSING)
    if ordinal_dot and stripped.endswith(".") and stripped[:-1].isdigit():
        return False
    return stripped.endswith(TERMINAL) and stripped.lower() not in abbreviations


def gaps(speech: list, duration: float, min_gap: float) -> list:
    """The VAD silences of at least min_gap, including the span's two ends, as (start, end)."""

    edges = [0.0] + [t for segment in speech for t in segment] + [duration]
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_gap]


def boundary(silences: list, before: float, after: float):
    """The longest silence between two words (before: the earlier word's end, after: the later's
    start), allowing TOLERANCE for Whisper's timing; None when they are not separated by a pause."""

    candidates = [(a, b) for a, b in silences
                  if a < after + TOLERANCE and b > before - TOLERANCE]
    return max(candidates, key=lambda gap: gap[1] - gap[0], default=None)


def pack(sentences: list, silences: list, args: argparse.Namespace) -> list:
    """Whole closed sentences packed into utterances, cut in real pauses; (begin, end, text)."""

    utterances = []
    index = 0
    while index < len(sentences):
        first = sentences[index]
        previous_end = sentences[index - 1].end if index else 0.0
        opening = boundary(silences, previous_end, first.start)
        # a sentence resumed after a long pause is not capitalised by Whisper; one that is starts
        # a Spanish question or exclamation opens on ¿/¡, a quote on its mark: the letter after decides
        initial = first.words[0].text.strip().lstrip(OPENING)[:1]
        if not first.closed or opening is None or not initial.isupper() and not initial.isdigit():
            index += 1
            continue

        # extend sentence by sentence; remember the last place the utterance could be closed
        best, last = None, index
        while last < len(sentences) and sentences[last].closed:
            sentence = sentences[last]
            if last > index and sentence.start - sentences[last - 1].end > args.max_pause:
                break
            if sentence.end - first.start > args.max_duration:
                break
            following = sentences[last + 1].start if last + 1 < len(sentences) else sentence.end + 1.0
            closing = boundary(silences, sentence.end, following)
            if closing is not None:
                length = sentence.end - first.start
                if length >= args.target_duration:
                    # close here or at the previous clean pause, whichever lands nearer the target
                    earlier = best and best[2] >= args.min_duration and \
                        args.target_duration - best[2] < length - args.target_duration
                    best = best if earlier else (last, closing, length)
                    break
                best = (last, closing, length)
            last += 1

        if best is None:
            index += 1
            continue
        last, closing, _ = best
        speech_start, speech_end = opening[1], closing[0]
        begin = speech_start - min(args.lead, opening[1] - opening[0])
        end = speech_end + min(args.trail, closing[1] - closing[0])
        if args.min_duration <= end - begin <= args.max_duration:
            text = "".join(word.text for sentence in sentences[index:last + 1]
                           for word in sentence.words).strip()
            utterances.append((begin, end, text))
        index = last + 1
    return utterances


def report(stats: dict, timings: dict, wall: float, args: argparse.Namespace) -> None:
    print_section("Done")
    print_info("recordings", f"{stats['recordings']} ({stats['skipped']} skipped)")
    print_info("source audio", f"{stats['source_hours']:.2f} h, of which {stats['read_hours']:.2f} h "
                               f"read ({stats['read_hours'] / max(stats['source_hours'], 1e-9):.0%})")
    print_info("utterances", f"{stats['utterances']}, {stats['utterance_hours']:.2f} h "
                             f"({stats['utterance_hours'] / max(stats['read_hours'], 1e-9):.0%} of read)")
    if stats["utterances"]:
        print_info("mean utterance", f"{stats['utterance_hours'] * 3600 / stats['utterances']:.1f} s")
    print_info("sentences", f"{stats['sentences']}, {stats['open_sentences']} left open")
    print_info("other language", f"{stats['foreign']} utterances, {stats['foreign_hours']:.2f} h "
                                 f"thrown away (Whisper detects a language other than {args.lang} in their window)")
    print_info("wall time", f"{wall:.0f} s, {stats['read_hours'] * 3600 / max(wall, 1e-9):.0f}x "
                            f"realtime over the read audio, "
                            f"{stats['source_hours'] * 3600 / max(wall, 1e-9):.0f}x over the source")
    print_info("stages", "read, resample and vad run in worker processes (summed over them); "
                         "waiting is the GPU loop idle for them")
    for stage, seconds in timings.items():
        print_info(f"  {stage}", f"{seconds:.0f} s ({seconds / max(wall, 1e-9):.0%})")
    print_info("csv", args.output)
    print_info("audio", args.audio_dir)


if __name__ == "__main__":
    main()
