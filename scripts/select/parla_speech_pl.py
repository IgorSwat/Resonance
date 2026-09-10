"""Select a prosodically varied subset of the filtered ParlaSpeech-PL CSV.

    python scripts/select/parla_speech_pl.py --min-per-speaker 3 --target-per-speaker 7

Takes the output of scripts/filter/parla_speech_pl.py as the input pool, scores every clip with
tools/prosody, and picks clips whose prosody-cell histogram is as flat as the pool allows. Every
speaker holding --min-per-speaker clips is admitted and contributes --target-per-speaker of them,
or all they have if fewer, then grows toward --max-per-speaker for as long as each further clip
still flattens the histogram, so the subset size is decided by the pool rather than fixed up
front. Unlike Emilia's diarisation labels, a speaker here is a named member of the Sejm, so the
per-speaker bounds act on a real person and the pool is deep per speaker.

The clips scored are OmniVoice's 24 kHz regenerations under --root, not the 16 kHz originals.

Male voices outnumber female ones roughly 3:2 in ParlaSpeech-PL, so high voices get their own
--high-target-per-speaker / --high-max-per-speaker. Which voices those are is read from the
speaker_gender column the filter wrote, rather than measured from the audio as it is for corpora
that carry no such label.
"""

import argparse
import csv
import pathlib
import random
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.select.libritts import capped_random, report, score
from tools.prosody.constants import DEFAULT_CEILING, DEFAULT_FLOOR, HIGH_TARGET
from tools.prosody.selection import select_bounded

DATASET = "ParlaSpeech-PL"
ROOT = pathlib.Path("data/processed/ParlaSpeech-PL/audio")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language", "speaker_gender")
SEPARATOR = "|"
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
REPORTED = ("st_range", "st_slope_std", "st_final_slope", "voiced_onsets_per_s",
            "voiced_frac", "n_phrases")
HIGH = "F"                      # the speaker_gender value treated as a high voice


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("parla_speech_pl_filtered.csv"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("parla_speech_pl_selected.csv"))
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="where the filter wrote its regenerations")
    parser.add_argument("--min-per-speaker", type=int, default=DEFAULT_FLOOR,
                        help="clips a speaker must have to be kept at all")
    parser.add_argument("--target-per-speaker", type=int, default=DEFAULT_FLOOR,
                        help="clips a kept speaker contributes, or all they have if fewer")
    parser.add_argument("--max-per-speaker", type=int, default=DEFAULT_CEILING,
                        help="most a speaker may contribute, reached only by flattening the histogram")
    parser.add_argument("--high-target-per-speaker", type=int, default=HIGH_TARGET,
                        help=f"clips a speaker labelled {HIGH} contributes, or all they have if fewer")
    parser.add_argument("--high-max-per-speaker", type=int, default=DEFAULT_CEILING,
                        help=f"as --max-per-speaker, for speakers labelled {HIGH}")
    parser.add_argument("--limit", type=int, help="score only N random rows of the input pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-accepted", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random selected clips to {ACCEPTED}/, "
                             "with a report of their prosody")
    return parser.parse_args()


def main():
    args = parse_args()
    pool, languages, genders = read_pool(args.input)
    if args.limit:
        pool = random.Random(args.seed).sample(pool, min(args.limit, len(pool)))

    print_test_title(f"Selecting from {DATASET}: {len(pool)} filtered clips")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("bounds", f"{args.target_per_speaker}-{args.max_per_speaker} clips per speaker  "
                         f"({args.high_target_per_speaker}-{args.high_max_per_speaker} "
                         f"for {HIGH})")
    print_info("admitted", f"speakers with at least {args.min_per_speaker} clips")

    paths = locate(args.root, {name for name, _, _ in pool})
    missing = [name for name, _, _ in pool if name not in paths]
    if missing:
        print(f"{Colors.WARNING}  {len(missing)} clips not found under {args.root}{Colors.ENDC}")

    start = time.time()
    rows, rejected = score(pool, paths)
    for row in rows:
        row["gender"] = genders[row["name"]]
    print_info("scored", f"{len(rows)} clips in {time.time() - start:.0f}s"
                         f"  (rejected {dict(rejected)})")
    if not rows:
        raise SystemExit("Nothing to select from")

    selected = select_bounded(rows, args.min_per_speaker, args.max_per_speaker, seed=args.seed,
                              target=args.target_per_speaker,
                              high_bounds=(args.high_target_per_speaker, args.high_max_per_speaker),
                              is_high=lambda clips: clips[0]["gender"] == HIGH)
    if not selected:
        raise SystemExit(f"No speaker has {args.min_per_speaker} usable clips")
    print_info("high-voiced", f"{high_share(rows):.1%} of the pool "
                              f"-> {high_share(selected):.1%} of the selection")
    write(args.output, selected, languages, genders)

    baseline = capped_random(rows, len(selected), args.max_per_speaker, args.seed)
    report(rows, selected, baseline, args.output)
    if args.dump_accepted:
        dump_accepted(selected, args.seed)


def high_share(rows):
    """Fraction of clips from a high voice — the imbalance the high bounds correct."""

    return sum(row["gender"] == HIGH for row in rows) / len(rows)


def read_pool(path):
    """The filtered CSV as (name, transcript, speaker) triples, plus its language and gender maps."""

    if not path.exists():
        raise SystemExit(f"No input pool at {path}; run scripts/filter/parla_speech_pl.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    return ([(row["name"], row["transcription"], row["speaker_id"]) for row in rows],
            {row["name"]: row["language"] for row in rows},
            {row["name"]: row["speaker_gender"] for row in rows})


def locate(root, names):
    """Map clip name to its wav, walking --root once rather than per clip."""

    return {path.stem: path for path in root.rglob("*.wav") if path.stem in names}


def write(path, selected, languages, genders):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, **DIALECT)
        writer.writerow(COLUMNS)
        for row in sorted(selected, key=lambda row: row["name"]):
            writer.writerow((DATASET, row["name"], row["transcript"], row["speaker"],
                             languages[row["name"]], genders[row["name"]]))


def dump_accepted(selected, seed):
    """Copy a listening sample next to what each clip measured."""

    ACCEPTED.mkdir(parents=True, exist_ok=True)
    for stale in ACCEPTED.iterdir():
        if stale.is_file():
            stale.unlink()

    sample = random.Random(seed).sample(selected, min(DUMP_SAMPLE, len(selected)))
    with open(ACCEPTED / "prosody.txt", "w") as report_file:
        for row in sorted(sample, key=lambda row: row["st_range"]):
            shutil.copy2(row["path"], ACCEPTED / f"range{row['st_range']:05.2f}_{row['name']}.wav")
            report_file.write(f"{row['name']}.wav  ({row['dur']:.2f} s, {row['gender']}, "
                              f"ref {row['ref_hz']:.0f} Hz, cell {row['cell']})\n  ")
            report_file.write("  ".join(f"{field} {row[field]:.2f}" for field in REPORTED))
            report_file.write("\n\n")
    print_info("dumped", f"{len(sample)} clips to {ACCEPTED}")


if __name__ == "__main__":
    main()
