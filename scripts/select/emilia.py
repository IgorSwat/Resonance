"""Select a prosodically varied subset of the filtered Emilia CSV.

    python scripts/select/emilia.py --min-per-speaker 7 --max-per-speaker 20

Takes the output of scripts/filter/emilia.py as the input pool, scores every clip with
tools/prosody, and picks clips whose prosody-cell histogram is as flat as the pool allows.
Every speaker with enough clips contributes --min-per-speaker of them, then grows toward
--max-per-speaker for as long as each further clip still flattens the histogram, so the subset
size is decided by the pool rather than fixed up front. Emilia speakers are per-source
diarisation labels, not people, so a speaker here is one voice in one recording and most of
them carry only a handful of clips; the comparison the script prints is what tells you whether
the selection actually moved anything.

Male voices outnumber female ones roughly 3:1 here, so speakers above PITCH_EDGE get their own
--high-min-per-speaker / --high-max-per-speaker instead, contributing more clips each.
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
from tools.prosody.constants import DEFAULT_CEILING, DEFAULT_FLOOR, HIGH_FLOOR, PITCH_EDGE
from tools.prosody.selection import select_bounded

DATASET = "Emilia"
ROOT = pathlib.Path("data/Emilia")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
SEPARATOR = "|"
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
REPORTED = ("st_range", "st_slope_std", "st_final_slope", "voiced_onsets_per_s",
            "voiced_frac", "n_phrases")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("emilia_filtered.csv"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("emilia_selected.csv"))
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--min-per-speaker", type=int, default=DEFAULT_FLOOR,
                        help="clips every kept speaker contributes; speakers with fewer are dropped")
    parser.add_argument("--max-per-speaker", type=int, default=DEFAULT_CEILING,
                        help="most a speaker may contribute, reached only by flattening the histogram")
    parser.add_argument("--high-min-per-speaker", type=int, default=HIGH_FLOOR,
                        help=f"as --min-per-speaker, for speakers above {PITCH_EDGE:.0f} Hz")
    parser.add_argument("--high-max-per-speaker", type=int, default=DEFAULT_CEILING,
                        help=f"as --max-per-speaker, for speakers above {PITCH_EDGE:.0f} Hz")
    parser.add_argument("--limit", type=int, help="score only N random rows of the input pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-accepted", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random selected clips to {ACCEPTED}/, "
                             "with a report of their prosody")
    return parser.parse_args()


def main():
    args = parse_args()
    pool, languages = read_pool(args.input)
    if args.limit:
        pool = random.Random(args.seed).sample(pool, min(args.limit, len(pool)))

    print_test_title(f"Selecting from {DATASET}: {len(pool)} filtered clips")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("bounds", f"{args.min_per_speaker}-{args.max_per_speaker} clips per speaker  "
                         f"({args.high_min_per_speaker}-{args.high_max_per_speaker} "
                         f"above {PITCH_EDGE:.0f} Hz)")

    paths = locate(args.root, {name for name, _, _ in pool})
    missing = [name for name, _, _ in pool if name not in paths]
    if missing:
        print(f"{Colors.WARNING}  {len(missing)} clips not found under {args.root}{Colors.ENDC}")

    start = time.time()
    rows, rejected = score(pool, paths)
    print_info("scored", f"{len(rows)} clips in {time.time() - start:.0f}s"
                         f"  (rejected {dict(rejected)})")
    if not rows:
        raise SystemExit("Nothing to select from")

    selected = select_bounded(rows, args.min_per_speaker, args.max_per_speaker, seed=args.seed,
                              high_bounds=(args.high_min_per_speaker, args.high_max_per_speaker))
    if not selected:
        raise SystemExit(f"No speaker has {args.min_per_speaker} usable clips")
    print_info("high-voiced", f"{high_share(rows):.1%} of the pool "
                              f"-> {high_share(selected):.1%} of the selection")
    write(args.output, selected, languages)

    baseline = capped_random(rows, len(selected), args.max_per_speaker, args.seed)
    report(rows, selected, baseline, args.output)
    if args.dump_accepted:
        dump_accepted(selected, args.seed)


def high_share(rows):
    """Fraction of clips spoken above PITCH_EDGE — the imbalance the high bounds correct."""

    return sum(row["ref_hz"] > PITCH_EDGE for row in rows) / len(rows)


def read_pool(path):
    """The filtered CSV, as (name, transcript, speaker) triples and a name -> language map."""

    if not path.exists():
        raise SystemExit(f"No input pool at {path}; run scripts/filter/emilia.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    return ([(row["name"], row["transcription"], row["speaker_id"]) for row in rows],
            {row["name"]: row["language"] for row in rows})


def locate(root, names):
    """Map clip name to its mp3, walking the batch directories once rather than per clip."""

    return {path.stem: path for path in root.rglob("*.mp3") if path.stem in names}


def write(path, selected, languages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, **DIALECT)
        writer.writerow(COLUMNS)
        for row in sorted(selected, key=lambda row: row["name"]):
            writer.writerow((DATASET, row["name"], row["transcript"], row["speaker"],
                             languages[row["name"]]))


def dump_accepted(selected, seed):
    """Copy a listening sample next to what each clip measured."""

    ACCEPTED.mkdir(parents=True, exist_ok=True)
    for stale in ACCEPTED.iterdir():
        if stale.is_file():
            stale.unlink()

    sample = random.Random(seed).sample(selected, min(DUMP_SAMPLE, len(selected)))
    with open(ACCEPTED / "prosody.txt", "w") as report_file:
        for row in sorted(sample, key=lambda row: row["st_range"]):
            shutil.copy2(row["path"], ACCEPTED / f"range{row['st_range']:05.2f}_{row['name']}.mp3")
            report_file.write(f"{row['name']}.mp3  ({row['dur']:.2f} s, "
                              f"ref {row['ref_hz']:.0f} Hz, cell {row['cell']})\n  ")
            report_file.write("  ".join(f"{field} {row[field]:.2f}" for field in REPORTED))
            report_file.write("\n\n")
    print_info("dumped", f"{len(sample)} clips to {ACCEPTED}")


if __name__ == "__main__":
    main()
