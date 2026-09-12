"""Filter WolneLektury through the quality cascade and write the survivors to a CSV.

    python scripts/filter/wolne_lektury.py --limit 100

WolneLektury ships as parquet shards (data/WolneLektury/train-00000-of-00381.parquet), each row
holding a clip inline beside its transcript, speaker and gender columns. Every shard under --root
is read in turn, one record batch at a time; --batch restricts the run to a single one.

The audio is the corpus's own 24 kHz recordings, so the cascade of tools/metrics/pipeline.py
applies as written and nothing is regenerated. Accepted clips are written out as wav, since a
parquet row cannot be reopened by path the way the later stages expect.

Use --limit while calibrating: it draws a random sample, so the rejection breakdown it prints
tells you whether the configured bounds keep a sensible share of the corpus before committing to
a full shard.
"""

import argparse
import collections
import csv
import io
import pathlib
import sys
import time

import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.filter.libritts import prepare, report
from scripts.filter.parla_speech_pl import sample
from tools.metrics.pipeline import Pipeline
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "WolneLektury"
ROOT = pathlib.Path("data/WolneLektury")
CONFIG = pathlib.Path("configurations/quality_filtering_wolne_lektury.yaml")
AUDIO = pathlib.Path("data/processed/WolneLektury/audio")
REJECTED = pathlib.Path("tmp/rejected")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100               # clips dumped per verdict; a clip cannot be re-read from disk
                                # later, so this caps as it streams rather than sampling
                                # uniformly the way the corpus filters do
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language", "speaker_gender")
FIELDS = ("__key__", "mp3", "text", "language", "speaker_id", "gender")
SEPARATOR = "|"
BATCH_ROWS = 64                 # record batch size; one clip's audio is ~450 kB


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a single shard under --root, e.g. train-00000-of-00381")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("wolne_lektury_filtered.csv"))
    parser.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO,
                        help="where the accepted clips are written as wav")
    parser.add_argument("--limit", type=int, help="process N random rows instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--dump", action="store_true",
                        help=f"copy up to {DUMP_SAMPLE} clips per verdict to {ACCEPTED}/ and "
                             f"{REJECTED}/, with a scores.txt in each giving the scores that "
                             "failed and the transcript")
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
    pipeline = Pipeline(config, verbose=args.verbose)
    print_info("stages", ", ".join(stage.verdict.value for stage in pipeline.stages))

    print_section("Filtering")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dumps = Dump(pipeline) if args.dump else None
    verdicts = collections.Counter()
    accepted, failures, start = [], 0, time.time()

    with open(args.output, "w", newline="") as handle:
        # Matches the other filters' dialect: no WolneLektury transcript contains SEPARATOR or a
        # backslash, so the writer never escapes, and quotechar=None leaves quotes as written.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        for index, row in enumerate(rows(shards)):
            if chosen is not None and index not in chosen:
                continue
            name = row["__key__"]
            try:
                audio = decode(row)
                verdict = pipeline.run(audio, row["text"])
            except Exception as error:
                print(f"{Colors.FAIL}{name}: {error}{Colors.ENDC}", flush=True)
                failures += 1
                continue

            verdicts[verdict] += 1
            done = sum(verdicts.values())
            duration = len(audio["audio"]) / audio["sample_rate"]
            targets = []
            if verdict is QualityVerdict.ACCEPTED:
                targets.append(args.audio_dir)
                writer.writerow((DATASET, name, row["text"], row["speaker_id"], row["language"],
                                 row["gender"]))
                handle.flush()
                accepted.append((row["speaker_id"], duration))
            if dumps:
                targets += dumps.record(name, audio, row["text"], verdict, duration)
            # dedupe, since --audio-dir may already be the directory the dump writes to
            for directory in dict.fromkeys(targets):
                sf.write(directory / f"{name}.wav", audio["audio"], audio["sample_rate"])
            if done % 200 == 0:
                rate = done / (time.time() - start)
                print(f"  {done}/{expected} clips, {rate:.1f}/s, "
                      f"{100 * len(accepted) / done:.0f}% accepted", flush=True)

    if dumps:
        dumps.close()
    report(verdicts, accepted, failures, time.time() - start, args.output)


def discover(args):
    """The shards to read, either the one named by --batch or every one under --root."""

    if args.batch:
        shard = (args.root / args.batch).with_suffix(".parquet")
        if not shard.exists():
            raise SystemExit(f"No shard {shard}")
        return [shard]
    shards = sorted(args.root.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No {DATASET} shards under {args.root}")
    return shards


def rows(shards):
    """Every row of every shard in order, a record batch at a time so a shard never fully loads."""

    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=BATCH_ROWS, columns=list(FIELDS)):
            yield from batch.to_pylist()


def decode(row):
    """The clip's audio. The 'mp3' column is misnamed: it holds 24 kHz PCM wav, not mp3."""

    audio, sample_rate = sf.read(io.BytesIO(row["mp3"]["bytes"]), dtype="float32")
    return {"audio": audio, "sample_rate": sample_rate}


class Dump:
    """Writes clips aside for listening, with the scores that judged them."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.directories = {True: prepare(ACCEPTED), False: prepare(REJECTED)}
        self.reports = {passed: open(directory / "scores.txt", "w")
                        for passed, directory in self.directories.items()}
        self.counts = collections.Counter()

    def record(self, name, audio, transcript, verdict, duration):
        """Note one clip; returns the directory its audio belongs in, or nothing once full."""

        passed = verdict is QualityVerdict.ACCEPTED
        if self.counts[passed] >= DUMP_SAMPLE:
            return []
        self.counts[passed] += 1

        report_file = self.reports[passed]
        report_file.write(f"{name}.wav  ({duration:.2f} s)\n")
        report_file.write(f"  verdict: {verdict.value}\n")
        for line in self.pipeline.describe(verdict, audio, transcript):
            report_file.write(line + "\n")
        report_file.write(f"  text: {transcript}\n\n")
        report_file.flush()
        return [self.directories[passed]]

    def close(self):
        for report_file in self.reports.values():
            report_file.close()
        for passed, directory in self.directories.items():
            print_info("dumped " + ("accepted" if passed else "rejected"),
                       f"{self.counts[passed]} -> {directory}")


if __name__ == "__main__":
    main()
