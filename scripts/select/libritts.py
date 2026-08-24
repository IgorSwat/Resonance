"""Select a prosodically varied subset of the filtered LibriTTS CSV.

    python scripts/select/libritts.py --per-speaker 7

Takes the output of scripts/filter/libritts.py as the input pool, scores every clip with
tools/prosody, and picks clips whose prosody-cell histogram is as flat as the pool allows,
targeting --per-speaker clips from each speaker. Random selection reproduces the source
distribution, which for audiobooks is mode-collapsed onto flat narration; the comparison the
script prints is what tells you whether the selection actually moved anything.
"""

import argparse
import collections
import csv
import pathlib
import random
import shutil
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import (Colors, print_info, print_section, print_separator,
                               print_summary_line, print_test_title)
from tools.prosody.constants import MIN_DURATION, SLOPE_LEVEL
from tools.prosody.contour import f0_contour
from tools.prosody.features import prosody_features
from tools.prosody.normalize import (adapted_range, clip_median, speaker_reference,
                                     to_semitones, within_octave_guard)
from tools.prosody.selection import assign_cells, select_diverse

DATASET = "LibriTTS"
ROOT = pathlib.Path("data/LibriTTS")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100
COLUMNS = ("dataset", "name", "transcription", "speaker_id")
SEPARATOR = "|"
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
REPORTED = ("st_range", "st_slope_std", "st_final_slope", "voiced_onsets_per_s",
            "voiced_frac", "n_phrases")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("libritts_filtered.csv"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("libritts_selected.csv"))
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--per-speaker", type=int, default=7,
                        help="target clips per speaker; the budget is this times the speaker count")
    parser.add_argument("--limit", type=int, help="score only N random rows of the input pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-accepted", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random selected clips to {ACCEPTED}/, "
                             "with a report of their prosody")
    return parser.parse_args()


def main():
    args = parse_args()
    pool = read_pool(args.input)
    if args.limit:
        pool = random.Random(args.seed).sample(pool, min(args.limit, len(pool)))

    print_test_title(f"Selecting from {DATASET}: {len(pool)} filtered clips")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("target", f"{args.per_speaker} clips per speaker")

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

    speakers = {row["speaker"] for row in rows}
    budget = args.per_speaker * len(speakers)
    selected = select_diverse(rows, budget, speaker_cap=args.per_speaker, seed=args.seed)
    write(args.output, selected)

    baseline = capped_random(rows, len(selected), args.per_speaker, args.seed)
    report(rows, selected, baseline, budget, args.output)
    if args.dump_accepted:
        dump_accepted(selected, args.seed)


def read_pool(path):
    """The filtered CSV, as (name, transcript, speaker) triples."""

    if not path.exists():
        raise SystemExit(f"No input pool at {path}; run scripts/filter/libritts.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    return [(row["name"], row["transcription"], row["speaker_id"]) for row in rows]


def locate(root, names):
    """Map clip name to its wav, walking the corpus once rather than per clip."""

    return {path.stem: path for path in root.rglob("*.wav") if path.stem in names}


def load(path):
    audio, sample_rate = sf.read(path, dtype="float32")
    return {"audio": audio.astype(np.float64), "sample_rate": sample_rate}


def score(pool, paths):
    """Two passes per speaker: a pitch reference, then a feature row relative to it."""

    by_speaker = collections.defaultdict(list)
    for name, transcript, speaker in pool:
        if name in paths:
            by_speaker[speaker].append((name, transcript))

    rows, rejected, done, start = [], collections.Counter(), 0, time.time()
    for speaker, clips in sorted(by_speaker.items()):
        medians = {}
        for name, _ in clips:
            audio = load(paths[name])
            if len(audio["audio"]) / audio["sample_rate"] < MIN_DURATION:
                rejected["too short"] += 1
                continue
            median = clip_median(f0_contour(audio))
            if median is None:
                rejected["unvoiced"] += 1
                continue
            medians[name] = median
        if not medians:
            continue

        reference = speaker_reference(list(medians.values()))
        floor, ceiling = adapted_range(reference)
        for name, transcript in clips:
            if name not in medians:
                continue
            if not within_octave_guard(medians[name], reference):
                rejected["octave guard"] += 1
                continue
            audio = load(paths[name])
            duration = len(audio["audio"]) / audio["sample_rate"]
            contour = to_semitones(f0_contour(audio, floor, ceiling), reference)
            features = prosody_features(contour, duration)
            if features is None:
                rejected["no features"] += 1
                continue
            rows.append(features | {"name": name, "transcript": transcript, "speaker": speaker,
                                    "path": paths[name], "ref_hz": reference, "dur": duration})
        done += len(clips)
        if len(rows) % 500 < len(clips):
            print(f"  {done} clips, {done / max(time.time() - start, 1e-9):.0f}/s", flush=True)
    return rows, rejected


def capped_random(rows, size, cap, seed):
    """Random selection under the same per-speaker cap — the baseline worth beating."""

    taken, picked = collections.Counter(), []
    for row in random.Random(seed).sample(rows, len(rows)):
        if len(picked) == size:
            break
        if taken[row["speaker"]] < cap:
            picked.append(row)
            taken[row["speaker"]] += 1
    return picked


def write(path, selected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, **DIALECT)
        writer.writerow(COLUMNS)
        for row in sorted(selected, key=lambda row: row["name"]):
            writer.writerow((DATASET, row["name"], row["transcript"], row["speaker"]))


def entropy(cells):
    counts = np.array(list(collections.Counter(cells).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def report(rows, selected, baseline, budget, output):
    for row, cell in zip(rows, assign_cells(rows)):        # pool edges, never a subset's own
        row["cell"] = cell
    n_cells = len({row["cell"] for row in rows})

    print_section("Selected")
    per_speaker = collections.Counter(row["speaker"] for row in selected)
    seconds = [row["dur"] for row in selected]
    minutes = sorted(sum(row["dur"] for row in selected if row["speaker"] == speaker)
                     for speaker in per_speaker)
    print_summary_line("  clips", f"{len(selected)} of {len(rows)} "
                                  f"({Colors.OKGREEN}{100 * len(selected) / len(rows):.1f}%{Colors.ENDC})"
                                  f", target {budget}")
    print_info("total length", f"{sum(seconds) / 3600:.2f} h")
    print_info("average length", f"{sum(seconds) / len(seconds):.2f} s")
    print_info("speakers", len(per_speaker))
    print_info("clips per speaker", f"min {min(per_speaker.values())}   "
                                    f"max {max(per_speaker.values())}")
    print_info("minutes per speaker", f"min {minutes[0] / 60:.2f}   max {minutes[-1] / 60:.2f}")

    print_section("Prosody — does the selection beat random?")
    print(f"  {'':10}{'cells':>10}{'entropy':>10}{'st_range':>18}{'rising':>9}{'dur':>8}")
    for label, subset in (("pool", rows), ("random", baseline), ("diversity", selected)):
        spread = np.array([row["st_range"] for row in subset])
        rising = np.mean([row["st_final_slope"] > SLOPE_LEVEL for row in subset])
        cells = len({row["cell"] for row in subset})
        colour = Colors.OKGREEN if label == "diversity" else Colors.ENDC
        print(f"  {colour}{label:10}{cells:6d}/{n_cells:<3d}{entropy([r['cell'] for r in subset]):10.2f}"
              f"{spread.mean():11.2f} +/-{spread.std():5.2f}{rising:9.2f}"
              f"{np.mean([r['dur'] for r in subset]):7.1f}s{Colors.ENDC}")
    print(f"  {'':10}{'':10}max {np.log2(n_cells):.2f}")

    print_separator()
    print_info("written to", output)


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
            report_file.write(f"{row['name']}.wav  ({row['dur']:.2f} s, "
                              f"ref {row['ref_hz']:.0f} Hz, cell {row['cell']})\n  ")
            report_file.write("  ".join(f"{field} {row[field]:.2f}" for field in REPORTED))
            report_file.write("\n\n")
    print_info("dumped", f"{len(sample)} clips to {ACCEPTED}")


if __name__ == "__main__":
    main()
