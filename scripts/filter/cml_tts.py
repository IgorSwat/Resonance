"""Filter CML-TTS through the quality cascade and write the survivors to a CSV.

    python scripts/filter/cml_tts.py --lang PL --limit 100

CML-TTS is one parquet directory per language (data/CML-TTS/polish/train-00000-of-00012-*.parquet),
each row holding a clip inline beside its transcript and speaker columns. Every shard for --lang
under --root is read in turn, one record batch at a time; --batch restricts the run to a single
one.

The audio is 24 kHz, so the cascade of tools/metrics/pipeline.py applies as written and nothing
is regenerated. Only the CSV is written: the shards already hold every clip, so
scripts/finalize/cml_tts.py reads the selected ones back out of them rather than have this script
lay down a second copy of the corpus. --dump is the exception, and it is capped.

The corpus ships a `duration` column, which this script ignores — it is computed at 22.05 kHz
against audio stored at 24 kHz, so every value is 8.8% too long (knowledge/cml_tts.md).

Use --limit while calibrating: it draws a random sample, so the rejection breakdown it prints
tells you whether the configured bounds keep a sensible share of the corpus before committing to
a full shard.
"""

import argparse
import collections
import csv
import io
import itertools
import multiprocessing
import pathlib
import sys
import time

import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.fetch.cml_tts import LANGUAGES
from scripts.filter.libritts import prepare, report
from scripts.filter.parla_speech_pl import sample
from tools.metrics.pipeline import Pipeline
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "CML-TTS"
ROOT = pathlib.Path("data/CML-TTS")
CONFIG = pathlib.Path("configurations/quality_filtering_cml_tts.yaml")
REJECTED = pathlib.Path("tmp/rejected")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100               # clips dumped per verdict; a clip cannot be re-read from disk
                                # later, so this caps as it streams rather than sampling
                                # uniformly the way the corpus filters do
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
FIELDS = ("audio", "text", "speaker_id")
SEPARATOR = "|"
BATCH_ROWS = 64                 # record batch size; one clip's audio is ~600 kB
WINDOW_PER_WORKER = 8           # tasks queued per worker; see scored()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="PL", type=str.upper, choices=sorted(LANGUAGES))
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a single shard under --root/<language>, without .parquet")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("cml_tts_filtered.csv"))
    parser.add_argument("--limit", type=int, help="process N random rows instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--dump", action="store_true",
                        help=f"copy up to {DUMP_SAMPLE} clips per verdict to {ACCEPTED}/ and "
                             f"{REJECTED}/, with a scores.txt in each giving the scores that "
                             "failed and the transcript")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel processes; each loads its own copy of the models, so more "
                             "is not faster — 8 measured slower than 1 and ran out of memory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    language = args.lang.lower()
    shards = discover(args)
    chosen, total = sample(shards, args)
    expected = len(chosen) if chosen else total

    print_test_title(f"Filtering {DATASET} {args.lang}: {expected} clips")
    print_info("root", args.root / LANGUAGES[args.lang])
    print_info("shards", args.batch or f"all {len(shards)}")
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("workers", args.workers)

    print_section("Filtering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dumps = Dump(config) if args.dump else None
    verdicts = collections.Counter()
    accepted, failures, start = [], 0, time.time()

    with open(args.output, "w", newline="") as handle:
        # Matches the other filters' dialect: no CML-TTS transcript contains SEPARATOR or a
        # backslash, so the writer never escapes, and quotechar=None leaves quotes as written.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        for row, verdict, error, duration in scored(wanted(shards, chosen), config, args):
            name = clip_name(row)
            if error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}", flush=True)
                failures += 1
                continue

            verdicts[verdict] += 1
            done = sum(verdicts.values())
            speaker = str(row["speaker_id"])
            if verdict is QualityVerdict.ACCEPTED:
                writer.writerow((DATASET, name, row["text"], speaker, language))
                handle.flush()
                accepted.append((speaker, duration))
            if dumps:
                dumps.record(row, verdict, duration)
            if done % 200 == 0:
                rate = done / (time.time() - start)
                print(f"  {done}/{expected} clips, {rate:.1f}/s, "
                      f"{100 * len(accepted) / done:.0f}% accepted", flush=True)

    if dumps:
        dumps.close()
    report(verdicts, accepted, failures, time.time() - start, args.output)


def discover(args):
    """The shards to read, either the one named by --batch or every one for --lang."""

    directory = args.root / LANGUAGES[args.lang]
    if args.batch:
        shard = (directory / args.batch).with_suffix(".parquet")
        if not shard.exists():
            raise SystemExit(f"No shard {shard}")
        return [shard]
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No {DATASET} shards under {directory}")
    return shards


def rows(shards):
    """Every row of every shard in order, a record batch at a time so a shard never fully loads."""

    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=BATCH_ROWS, columns=list(FIELDS)):
            yield from batch.to_pylist()


def clip_name(row):
    """'5090_1447_000098.wav' -> '5090_1447_000098'; the only per-clip id the corpus carries."""

    return row["audio"]["path"].rsplit(".", 1)[0]


def decode(row):
    """The clip's audio, 24 kHz wav inline in the parquet row."""

    audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    return {"audio": audio, "sample_rate": sample_rate}


def wanted(shards, chosen):
    """The rows to score: every one, or only those --limit sampled."""

    for index, row in enumerate(rows(shards)):
        if chosen is None or index in chosen:
            yield row


def scored(stream, config, args):
    """Verdicts for every clip, in a worker pool unless --workers 1.

    The file-based filters hand their workers a path and let each one open its own clip. A row
    here already carries the audio, ~600 kB of it, and Pool's task thread drains whatever
    iterable it is given as fast as it can — so feeding it the stream directly would pull the
    whole shard into memory. The stream is passed a window at a time instead, which bounds what
    is in flight to WINDOW_PER_WORKER clips per worker.

    Only the verdict comes back; the audio stays in the parent, which already holds the window.
    """

    if args.workers <= 1:
        _setup(config, args.verbose)
        for row in stream:
            yield (row, *_score(row)[1:])
        return

    size = WINDOW_PER_WORKER * args.workers
    with multiprocessing.Pool(args.workers, _setup, (config, args.verbose)) as pool:
        while window := list(itertools.islice(stream, size)):
            sources = {clip_name(row): row for row in window}
            for name, verdict, error, duration in pool.imap_unordered(_score, window, chunksize=1):
                yield sources[name], verdict, error, duration


_pipeline = None


def _setup(config, verbose):
    """One pipeline per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline
    torch.set_num_threads(1)
    _pipeline = Pipeline(config, verbose=verbose)


def _score(row):
    """One clip through the cascade, named rather than returned, so no audio is pickled back."""

    name = clip_name(row)
    if not row["text"]:
        return name, None, "no transcript", 0.0
    try:
        audio = decode(row)
    except Exception as error:
        return name, None, str(error), 0.0
    verdict = _pipeline.run(audio, row["text"])
    return name, verdict, None, len(audio["audio"]) / audio["sample_rate"]


class Dump:
    """Writes clips aside for listening, with the scores that judged them."""

    def __init__(self, config):
        self.pipeline = Pipeline(config)
        self.directories = {True: prepare(ACCEPTED), False: prepare(REJECTED)}
        self.reports = {passed: open(directory / "scores.txt", "w")
                        for passed, directory in self.directories.items()}
        self.counts = collections.Counter()

    def record(self, row, verdict, duration):
        """Copy one clip aside with the scores that judged it, until this verdict is full.

        The audio is decoded here rather than passed in, so a full dump costs nothing.
        """

        passed = verdict is QualityVerdict.ACCEPTED
        if self.counts[passed] >= DUMP_SAMPLE:
            return
        self.counts[passed] += 1

        name, transcript, audio = clip_name(row), row["text"], decode(row)
        sf.write(self.directories[passed] / f"{name}.wav", audio["audio"], audio["sample_rate"])
        report_file = self.reports[passed]
        report_file.write(f"{name}.wav  ({duration:.2f} s)\n")
        report_file.write(f"  verdict: {verdict.value}\n")
        for line in self.pipeline.describe(verdict, audio, transcript):
            report_file.write(line + "\n")
        report_file.write(f"  text: {transcript}\n\n")
        report_file.flush()

    def close(self):
        for report_file in self.reports.values():
            report_file.close()
        for passed, directory in self.directories.items():
            print_info("dumped " + ("accepted" if passed else "rejected"),
                       f"{self.counts[passed]} -> {directory}")


if __name__ == "__main__":
    main()
