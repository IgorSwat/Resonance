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

**Whisper runs last on purpose.** It is the only stage that costs a GPU transcription, so the
DSP, NISQA and diarisation stages run first and roughly half the utterances are gone before a
transcript is ever asked for. The CTC stage then aligns the clip against what Whisper heard
rather than against the caption.

Unlike the parquet corpora this one writes its audio out: an utterance is a cut from a long-form
file and has no addressable existence anywhere else, so --audio-dir holds the accepted clips and
the later stages read them from there.
"""

import argparse
import collections
import csv
import dataclasses
import json
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
WHISPER_RATE = 16000            # both backends assume a 16 kHz mono float32 array
CUDA_MODEL = "large-v3-turbo"                            # faster-whisper's own name
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"       # the same weights, Metal build
PROGRESS_EVERY = 200


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
    parser.add_argument("--limit", type=int,
                        help="process N random recordings instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--whisper", help=f"model override (default: {CUDA_MODEL} on CUDA, "
                                          f"{MLX_MODEL} on an Apple GPU)")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    return parser.parse_args()


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    recordings = discover(args)

    print_test_title(f"Filtering {DATASET}: {len(recordings)} recordings")
    print_info("root", args.root)
    print_info("batch", args.batch or f"all {len({p.parent for p in recordings})}")
    print_info("chunks", f"captions joined across gaps up to {args.max_gap:.1f} s, "
                         f"closed at {args.target_duration:.1f} s")
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("audio", args.audio_dir)

    print_section("Loading models")
    # The cheap stages run without a transcript, so the cascade is built with CTC switched off and
    # the aligner is held separately for the clips that earn one.
    pipeline = Pipeline(dataclasses.replace(config, ctc_enabled=False), verbose=args.verbose)
    print_info("cascade", ", ".join(stage.verdict.value for stage in pipeline.stages))
    ctc = CtcAlignmentMetric(device=config.ctc_device, uroman=config.ctc_uroman_enabled)
    whisper = Whisper(args.whisper, args.lang)
    print_info("whisper", f"{whisper.name} ({whisper.backend})")
    print_info("ctc gate", "on" if config.ctc_enabled else "off (transcript kept, not gated)")

    print_section("Filtering")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verdicts = collections.Counter()
    accepted, failures, transcribed, start = [], 0, 0, time.time()

    with open(args.output, "w", newline="") as handle:
        # Whisper emits no SEPARATOR and no backslash, so the writer never escapes; quotechar=None
        # leaves the quotes it does emit as written, matching the other filters' dialect.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        for name, video, audio in utterances(recordings, args.max_gap, args.target_duration,
                                             config.max_duration):
            try:
                verdict = pipeline.run(audio)
                text = None
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
            except Exception as error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}", flush=True)
                failures += 1
                continue

            verdicts[verdict] += 1
            done = sum(verdicts.values())
            if verdict is QualityVerdict.ACCEPTED:
                sf.write(args.audio_dir / f"{name}.wav", audio["audio"], audio["sample_rate"])
                writer.writerow((DATASET, name, text, video, args.lang))
                handle.flush()
                accepted.append((video, len(audio["audio"]) / audio["sample_rate"]))
            if done % PROGRESS_EVERY == 0:
                rate = done / (time.time() - start)
                print(f"  {done} utterances, {rate:.1f}/s, {transcribed} transcribed, "
                      f"{100 * len(accepted) / done:.0f}% accepted", flush=True)

    print_info("whisper calls", f"{transcribed} of {sum(verdicts.values())} utterances "
                                f"({100 * transcribed / max(sum(verdicts.values()), 1):.0f}%)")
    report(verdicts, accepted, failures, time.time() - start, args.output)


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
        spans = meta.get("utterances") or {}
        chunks = merge(list(zip(spans.get("utt_id", []), spans.get("start", []),
                                spans.get("end", []))), max_gap, target, limit)
        with sf.SoundFile(flac) as handle:
            rate = handle.samplerate
            for name, begin, end in chunks:
                handle.seek(int(begin * rate))
                y = handle.read(int((end - begin) * rate), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                yield name, meta["video_id"], {"audio": y, "sample_rate": rate}


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

        y = to_whisper_rate(audio)
        if self.mlx is not None:
            spoken = self.mlx.transcribe(y, path_or_hf_repo=self.name, language=self.language,
                                         verbose=None)["text"]
        else:
            segments, _ = self.model.transcribe(y, language=self.language)
            spoken = " ".join(segment.text for segment in segments)
        return " ".join(spoken.split())


def to_whisper_rate(audio):
    """Both backends assume 16 kHz; the corpus is 24 kHz. Same kaiser parameters as the codec."""

    y = np.asarray(audio["audio"], dtype=np.float32)
    if audio["sample_rate"] == WHISPER_RATE:
        return y
    return torchaudio.functional.resample(torch.from_numpy(y), audio["sample_rate"],
                                          WHISPER_RATE, **RESAMPLE).numpy()


if __name__ == "__main__":
    main()
