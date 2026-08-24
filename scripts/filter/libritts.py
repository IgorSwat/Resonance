"""Filter LibriTTS through the quality cascade and write the survivors to a CSV.

    python scripts/filter/libritts.py --limit 500

Use --limit while calibrating: it draws a random sample, so the rejection breakdown it prints
tells you whether the configured bounds keep a sensible share of the corpus before committing
to a full pass (149,736 clips at ~175 ms each is several hours).
"""

import argparse
import collections
import csv
import multiprocessing
import pathlib
import random
import shutil
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import (Colors, print_info, print_section, print_separator,
                               print_summary_line, print_test_title)
from tools.metrics.nisqa import FIELDS as NISQA_FIELDS, NisqaMetric
from tools.metrics.pipeline import Pipeline
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "LibriTTS"
ROOT = pathlib.Path("data/LibriTTS")
CONFIG = pathlib.Path("configurations/quality_filtering.yaml")
REJECTED = pathlib.Path("tmp/rejected")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100                       # clips copied for listening, drawn uniformly
COLUMNS = ("dataset", "name", "transcription", "speaker_id")
SEPARATOR = "|"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("libritts_filtered.csv"))
    parser.add_argument("--limit", type=int, help="process N random files instead of the whole set")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--dump-rejected", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random clips rejected on quality (not length) "
                             f"to {REJECTED}/, with a report of the scores that failed")
    parser.add_argument("--dump-accepted", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random accepted clips to {ACCEPTED}/, "
                             "with a report of their NISQA scores")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel processes; each loads its own copy of the models, so more "
                             "is not faster — 8 measured slower than 1 and ran out of memory")
    return parser.parse_args()


_pipeline = None


def _setup(config, verbose):
    """One pipeline per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline
    torch.set_num_threads(1)
    _pipeline = Pipeline(config, verbose=verbose)


def _score(path):
    name = path.name.removesuffix(".normalized.txt")
    try:
        audio, transcript = load(path, name)
    except Exception as error:
        return path, None, str(error), 0.0, ""
    verdict = _pipeline.run(audio, transcript)
    return path, verdict, None, len(audio["audio"]) / audio["sample_rate"], transcript


class Reservoir:
    """Uniform sample of a stream whose length is not known in advance (algorithm R)."""

    def __init__(self, size, seed):
        self.size = size
        self.random = random.Random(seed)
        self.seen = 0
        self.items = []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        index = self.random.randrange(self.seen)
        if index < self.size:
            self.items[index] = item


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()

    transcripts = sorted(args.root.rglob("*.normalized.txt"))
    if not transcripts:
        raise SystemExit(f"No LibriTTS transcripts under {args.root}")
    if args.limit:
        transcripts = random.Random(args.seed).sample(transcripts, min(args.limit, len(transcripts)))

    print_test_title(f"Filtering {DATASET}: {len(transcripts)} clips")
    print_info("root", args.root)
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)

    print_info("workers", args.workers)
    if args.dump_rejected:
        print_info("rejected sample", f"{DUMP_SAMPLE} clips -> {REJECTED}")
    if args.dump_accepted:
        print_info("accepted sample", f"{DUMP_SAMPLE} clips -> {ACCEPTED}")

    rejected_sample = Reservoir(DUMP_SAMPLE, args.seed)
    accepted_sample = Reservoir(DUMP_SAMPLE, args.seed)

    verdicts = collections.Counter()
    accepted, failures, start = [], 0, time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        # 20% of LibriTTS transcripts open with a dialogue quote. quotechar=None leaves those
        # quotes as written instead of wrapping the field or escaping them; the escapechar is
        # only ever used if a transcript contains SEPARATOR, which none in this corpus does.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        for index, (path, verdict, error, duration, transcript) in enumerate(
            scored(transcripts, config, args), 1
        ):
            name = path.name.removesuffix(".normalized.txt")
            if error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}")
                failures += 1
                continue

            verdicts[verdict] += 1
            if verdict is QualityVerdict.ACCEPTED:
                writer.writerow((DATASET, name, transcript, name.split("_")[0]))
                handle.flush()
                accepted.append((name.split("_")[0], duration))
                accepted_sample.add((path, duration))
            elif verdict not in (QualityVerdict.TOO_SHORT, QualityVerdict.TOO_LONG):
                # length is a sourcing decision rather than a defect, so those are never dumped
                rejected_sample.add((path, verdict, duration))
            if index % 200 == 0:
                rate = index / (time.time() - start)
                print(f"  {index}/{len(transcripts)} clips, {rate:.1f}/s, "
                      f"{100 * len(accepted) / index:.0f}% accepted", flush=True)

    if args.dump_rejected:
        dump_rejected(rejected_sample, config)
    if args.dump_accepted:
        dump_accepted(accepted_sample, config)
    report(verdicts, accepted, failures, time.time() - start, args.output)


def prepare(directory):
    """A dump directory holding only this run's clips, so stale ones cannot mislead."""

    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.iterdir():
        if stale.is_file():
            stale.unlink()
    return directory


def dump_rejected(sample, config):
    """Copy each sampled clip next to a note of what it scored and what was allowed.

    The scores are recomputed here, for the sample only — running describe() on every rejected
    clip would re-evaluate a metric the pipeline had already decided on.
    """

    prepare(REJECTED)
    pipeline = Pipeline(config)
    with open(REJECTED / "rejections.txt", "w") as report_file:
        for path, verdict, duration in sorted(sample.items, key=lambda item: item[1].value):
            name = path.name.removesuffix(".normalized.txt")
            audio, transcript = load(path, name)
            shutil.copy2(path.with_name(f"{name}.wav"), REJECTED / f"{verdict.value}_{name}.wav")
            report_file.write(f"{name}.wav  ({duration:.2f} s)\n")
            report_file.write(f"  rejected by: {verdict.value}\n")
            report_file.write("\n".join(pipeline.describe(verdict, audio, transcript)) + "\n\n")
    print_info("rejected dumped", f"{len(sample.items)} of {sample.seen} to {REJECTED}")


def dump_accepted(sample, config):
    """Copy each sampled clip and record what NISQA thought of it."""

    prepare(ACCEPTED)
    nisqa = NisqaMetric(min_duration=config.min_duration, max_duration=config.nisqa_max_duration)
    with open(ACCEPTED / "nisqa.txt", "w") as report_file:
        for path, duration in sorted(sample.items):
            name = path.name.removesuffix(".normalized.txt")
            audio, _ = load(path, name)
            shutil.copy2(path.with_name(f"{name}.wav"), ACCEPTED / f"{name}.wav")
            scores = nisqa.evaluate(audio)
            report_file.write(f"{name}.wav  ({duration:.2f} s)\n  ")
            report_file.write("  ".join(f"{field} {scores[field]:.2f}" for field in NISQA_FIELDS))
            report_file.write("\n\n")
    print_info("accepted dumped", f"{len(sample.items)} of {sample.seen} to {ACCEPTED}")


def scored(transcripts, config, args):
    """Verdicts for every clip, in a worker pool unless --workers 1."""

    if args.workers <= 1:
        _setup(config, args.verbose)
        yield from (_score(path) for path in transcripts)
        return
    with multiprocessing.Pool(args.workers, _setup, (config, args.verbose)) as pool:
        yield from pool.imap_unordered(_score, transcripts, chunksize=8)


def load(path, name):
    audio, sample_rate = sf.read(path.with_name(f"{name}.wav"), dtype="float32")
    return {"audio": audio, "sample_rate": sample_rate}, path.read_text().strip()


def report(verdicts, accepted, failures, elapsed, output):
    total = sum(verdicts.values()) + failures
    print_section("Rejections")
    rejected = sum(count for verdict, count in verdicts.items() if verdict is not QualityVerdict.ACCEPTED)
    for verdict, count in sorted(verdicts.items(), key=lambda item: -item[1]):
        if verdict is QualityVerdict.ACCEPTED:
            continue
        print(f"  {verdict.value:22} {count:7d}  {Colors.WARNING}{100 * count / total:5.1f}%{Colors.ENDC}")
    if failures:
        print(f"  {'unreadable':22} {failures:7d}  {Colors.FAIL}{100 * failures / total:5.1f}%{Colors.ENDC}")
    if not rejected and not failures:
        print("  none")

    print_section("Accepted")
    if not accepted:
        print(f"  {Colors.FAIL}nothing accepted{Colors.ENDC}")
        return

    seconds = [duration for _, duration in accepted]
    per_speaker = collections.Counter()
    for speaker, duration in accepted:
        per_speaker[speaker] += duration
    minutes = sorted(per_speaker.values())

    print_summary_line("  clips", f"{len(accepted)} of {total} "
                                  f"({Colors.OKGREEN}{100 * len(accepted) / total:.1f}%{Colors.ENDC})")
    print_info("total length", f"{sum(seconds) / 3600:.2f} h")
    print_info("average length", f"{sum(seconds) / len(seconds):.2f} s")
    print_info("speakers", len(per_speaker))
    print_info("minutes per speaker", f"min {minutes[0] / 60:.2f}   max {minutes[-1] / 60:.2f}")
    print_separator()
    print_info("elapsed", f"{elapsed / 60:.1f} min ({total / max(elapsed, 1e-9):.1f} clips/s)")
    print_info("written to", output)


if __name__ == "__main__":
    main()
