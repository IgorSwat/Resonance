"""Regenerate ParlaSpeech-PL with OmniVoice and keep the clips it reproduces faithfully.

    python scripts/filter/parla_speech_pl.py --limit 5

ParlaSpeech-PL ships as parquet shards (data/ParlaSpeech-PL/train-00000-of-00123.parquet), each
row holding a FLAC blob beside its transcript and speaker columns. Every shard under --root is
read in turn, one record batch at a time; --batch restricts the run to a single one.

The source is 16 kHz parliamentary audio, so it is not kept: each clip is fed to OmniVoice as its
own voice-clone reference with its own transcript as the target text, and what lands in
--audio-dir is the 24 kHz regeneration. That buys the codec's sample rate at the cost of a model
free to say something other than what it was asked, which is what the CTC stage is here to catch
— the regeneration is aligned against the target text and only a faithful one is written out.
Nothing else from the quality cascade applies: the audio is synthetic, so its noise floor,
bandwidth and hum describe the vocoder rather than the recording. Transcripts carrying a digit
are dropped before regeneration, since the model reads "45" as words the aligner then scores
against the digits as written; about a fifth of the corpus goes this way.

Regeneration dominates the runtime at roughly a second of compute per second of audio, so
calibrate with --limit before committing to a shard.
"""

import argparse
import collections
import csv
import io
import pathlib
import random
import re
import sys
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.filter.libritts import prepare, report
from tools.metrics.ctc_alignment_metric import (CtcAlignmentMetric, DEFAULT_RBOUND,
                                                FIELDS as CTC_FIELDS, missing_words,
                                                redundant_speech)
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "ParlaSpeech-PL"
ROOT = pathlib.Path("data/ParlaSpeech-PL")
CONFIG = pathlib.Path("configurations/quality_filtering_parla_speech_pl.yaml")
AUDIO = pathlib.Path("data/processed/ParlaSpeech-PL/audio")
REJECTED = pathlib.Path("tmp/rejected")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100               # clips dumped per verdict; a generated clip cannot be re-read
                                # from disk later, so this caps as it streams rather than
                                # sampling uniformly the way the corpus filters do
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language", "speaker_gender")
FIELDS = ("id", "audio", "text", "speaker_name", "speaker_gender")
LANGUAGE = "pl"
SEPARATOR = "|"
BATCH_ROWS = 64                 # record batch size; one clip's audio is ~200 kB decoded
PROGRESS_EVERY = 10             # clips between the running rate/ETA lines
NUM_STEP = 32
T_SHIFT = 0.1
GENERATED_RATE = 24000          # OmniVoice's output rate, fixed by its codec
REFERENCE_PAD = 0.3             # silence appended to the reference, seconds. ParlaSpeech clips
                                # are cut tight to the utterance, leaving no trailing silence
                                # to mark where it ended.
DIGIT = re.compile(r"\d")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a single shard under --root, e.g. train-00122-of-00123")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("parla_speech_pl_filtered.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO,
                        help="where the accepted regenerations are written as 24 kHz wav")
    parser.add_argument("--limit", type=int, help="process N random rows instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true",
                        help="add each clip's CTC scores to its progress line")
    parser.add_argument("--dump", action="store_true",
                        help=f"copy up to {DUMP_SAMPLE} regenerations per verdict to {ACCEPTED}/ "
                             f"and {REJECTED}/, with a scores.txt in each giving the CTC scores, "
                             "the target text and what rejected it")
    return parser.parse_args()


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    shards = discover(args)
    chosen, total = sample(shards, args)
    expected = len(chosen) if chosen else total

    print_test_title(f"Filtering {DATASET}: {expected} clips")
    print_info("root", args.root)
    print_info("shards", args.batch or f"all {len(shards)}")
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("audio", args.audio_dir)

    print_section("Loading models")
    # fp16 segfaults the MPS backend while loading the language model; fp32 is the only dtype
    # that loads there, and the codec forces fp32 for itself regardless of what it is given.
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device(), dtype=torch.float32)
    ctc = CtcAlignmentMetric(device=config.ctc_device, uroman=config.ctc_uroman_enabled)
    bounds = {**DEFAULT_RBOUND, **config.ctc_max}
    print_info("omnivoice", f"{device()}, {NUM_STEP} steps, t_shift {T_SHIFT}")

    print_section("Regenerating")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dumps = Dump(bounds) if args.dump else None
    verdicts = collections.Counter()
    accepted, failures, start = [], 0, time.time()

    with open(args.output, "w", newline="") as handle:
        # Matches the other filters' dialect: no ParlaSpeech transcript contains SEPARATOR or a
        # backslash, so the writer never escapes, and quotechar=None leaves quotes as written.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        for index, row in enumerate(rows(shards)):
            if chosen is not None and index not in chosen:
                continue
            name = row["id"]
            try:
                verdict, duration, generated, scores = judge(row, model, ctc, bounds, config)
            except Exception as error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}", flush=True)
                failures += 1
                continue

            verdicts[verdict] += 1
            done = sum(verdicts.values())
            speaker = speaker_id(row["speaker_name"])
            targets = []
            if verdict is QualityVerdict.ACCEPTED:
                targets.append(args.audio_dir)
                writer.writerow((DATASET, name, row["text"], speaker, LANGUAGE,
                                 row["speaker_gender"]))
                handle.flush()
                accepted.append((speaker, duration))
            progress(done, expected, name, verdict, duration, scores, len(accepted), start,
                     args.verbose)
            if dumps:
                targets += dumps.record(name, row["text"], verdict, duration, scores)
            # dedupe, since --audio-dir may already be the directory the dump writes to
            for directory in dict.fromkeys(targets):
                if generated is not None:
                    sf.write(directory / f"{name}.wav", generated, GENERATED_RATE)

    if dumps:
        dumps.close()
    report(verdicts, accepted, failures, time.time() - start, args.output)


def progress(done, expected, name, verdict, duration, scores, accepted, start, verbose):
    """One line per clip, plus a rate and ETA every PROGRESS_EVERY.

    Regeneration takes seconds per clip, so the per-200 summary the corpus filters print would
    leave this one silent for half an hour at a time.
    """

    passed = verdict is QualityVerdict.ACCEPTED
    colour = Colors.OKGREEN if passed else Colors.WARNING
    line = (f"  {done:>{len(str(expected))}}/{expected}  "
            f"{colour}{verdict.value:<14}{Colors.ENDC}  {duration:5.2f}s  {name}")
    if verbose and scores:
        line += "   " + "  ".join(f"{field} {scores[field]:.3f}" for field in CTC_FIELDS)
    print(line, flush=True)

    if done % PROGRESS_EVERY:
        return
    rate = done / max(time.time() - start, 1e-9)
    print_info("progress", f"{done}/{expected}  {100 * accepted / done:.0f}% accepted  "
                           f"{rate:.2f} clips/s  ~{(expected - done) / max(rate, 1e-9) / 60:.0f} "
                           "min left")


def device():
    """Where OmniVoice runs; its codec probes the same device and falls back on its own."""

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def discover(args):
    """The shards to read, either the one named by --batch or every one under --root."""

    if args.batch:
        shard = (args.root / args.batch).with_suffix(".parquet")
        if not shard.exists():
            raise SystemExit(f"No shard {shard}")
        return [shard]
    shards = sorted(args.root.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No ParlaSpeech-PL shards under {args.root}")
    return shards


def sample(shards, args):
    """Row indices to keep for --limit, read from the parquet footers without touching audio."""

    total = sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)
    if not args.limit or args.limit >= total:
        return None, total
    return set(random.Random(args.seed).sample(range(total), args.limit)), total


def rows(shards):
    """Every row of every shard in order, a record batch at a time so a shard never fully loads."""

    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=BATCH_ROWS, columns=list(FIELDS)):
            yield from batch.to_pylist()


def judge(row, model, ctc, bounds, config):
    """Regenerate one clip and score it; returns verdict, duration, audio and its CTC scores.

    Scores come from evaluate() rather than validate() so the dump can report the numbers the
    decision was made on; the two bound checks below are what validate() applies internally.
    """

    source, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    duration = len(source) / rate
    if duration < config.min_duration:
        return QualityVerdict.TOO_SHORT, duration, None, None
    if duration > config.max_duration:
        return QualityVerdict.TOO_LONG, duration, None, None

    transcript = row["text"]
    if config.reject_digits_enabled and DIGIT.search(transcript):
        return QualityVerdict.DIGITS, duration, None, None
    # padded for the reference only; duration above is the clip's own, as the CSV reports it
    reference = np.concatenate([source, np.zeros(round(REFERENCE_PAD * rate), source.dtype)])
    generated = model.generate(text=transcript, ref_audio=(reference, rate), ref_text=transcript,
                               num_step=NUM_STEP, t_shift=T_SHIFT)[0]
    scores = ctc.evaluate({"audio": generated, "sample_rate": GENERATED_RATE}, transcript)
    if missing_words(scores, bounds) or redundant_speech(scores, bounds):
        return QualityVerdict.CTC_ALIGNMENT, duration, generated, scores
    return QualityVerdict.ACCEPTED, duration, generated, scores


class Dump:
    """Writes the regenerations aside for listening, with the scores that judged them."""

    def __init__(self, bounds):
        self.bounds = bounds
        self.directories = {True: prepare(ACCEPTED), False: prepare(REJECTED)}
        self.reports = {passed: open(directory / "scores.txt", "w")
                        for passed, directory in self.directories.items()}
        self.counts = collections.Counter()

    def record(self, name, transcript, verdict, duration, scores):
        """Note one clip; returns the directory its audio belongs in, or nothing once full."""

        passed = verdict is QualityVerdict.ACCEPTED
        if self.counts[passed] >= DUMP_SAMPLE:
            return []
        self.counts[passed] += 1

        report_file = self.reports[passed]
        report_file.write(f"{name}.wav  ({duration:.2f} s)\n")
        report_file.write(f"  verdict: {verdict.value}\n")
        if scores is None:
            report_file.write("  ctc: not scored — rejected on length before regenerating\n")
        else:
            report_file.write("  ctc: " + "  ".join(
                f"{field} {scores[field]:.3f} (max {self.bounds[field]})" for field in CTC_FIELDS
            ) + f"   [max_loss {self.bounds['max_loss']}]\n")
            report_file.write(f"  failed: {', '.join(self._failed(scores)) or 'nothing'}\n")
        report_file.write(f"  text: {transcript}\n\n")
        report_file.flush()
        return [self.directories[passed]]

    def _failed(self, scores):
        """Which of the two CTC defects fired, named as the metric names them."""

        return ([] if not missing_words(scores, self.bounds) else ["missing_words"]) + \
               ([] if not redundant_speech(scores, self.bounds) else ["redundant_speech"])

    def close(self):
        for report_file in self.reports.values():
            report_file.close()
        for passed, directory in self.directories.items():
            print_info("dumped " + ("accepted" if passed else "rejected"),
                       f"{self.counts[passed]} -> {directory}")


def speaker_id(name):
    """'Kuchciński, Marek' -> 'kuchcinski_marek'; MMS-safe ASCII, stable across shards."""

    folded = unicodedata.normalize("NFKD", name.replace("ł", "l").replace("Ł", "L"))
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return "_".join(stripped.lower().replace(",", " ").split())


if __name__ == "__main__":
    main()
