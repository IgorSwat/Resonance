"""Select the YODAS2-Sidon clips the student trails the teacher on most, within per-speaker bounds.

    python scripts/select/yodas2_sidon_dce.py --input en000_train-00000_scored.csv --share 0.33

Takes the CSV Echo's scripts/score_utterances.py writes from the output of
scripts/filter/yodas2_sidon.py and its speaker fixup — the same rows with a dce_score column — and
keeps --share of its clips,
taking those with the highest score subject to two bounds:

- every speaker holding at least --min-per-speaker clips keeps that many, their best ones; a
  speaker with fewer is left out, since it could not meet the floor;
- no speaker contributes more than --max-per-speaker.

The floors are taken first and the rest of the budget goes to the highest remaining scores under
the cap. With a per-speaker cap as the only constraint, taking the best clip still allowed is
optimal, so this maximizes the summed score for the fixed floors.

The raw score favours short clips: fewer scored tokens make it noisier, and the clamp in it turns
that noise into score. So by default it is ranked as a z-score within duration bands — the bands
of scripts/finalize/emilia_prune.py — which compares every clip only with clips of its own length.
"""

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import print_info, print_section, print_summary_line, print_test_title
from scripts.filter.yodas2_sidon import DATASET, SEPARATOR
from scripts.finalize.emilia_prune import BANDS

DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
SCORE = "dce_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, required=True,
                        help=f"a filtered CSV with a {SCORE} column")
    parser.add_argument("--output", type=pathlib.Path,
                        help="default: the input with _selected in place of its last suffix")
    parser.add_argument("--share", type=float, default=0.33, help="fraction of the clips to keep")
    parser.add_argument("--min-per-speaker", type=int, default=4,
                        help="clips every kept speaker gets; speakers with fewer are left out")
    parser.add_argument("--max-per-speaker", type=int, default=64)
    parser.add_argument("--raw", action="store_true",
                        help="rank by the raw score rather than by its z within duration bands")
    parser.add_argument("--random", action="store_true",
                        help=f"rank in a seeded random order instead, for a CSV without {SCORE}; with --share 1.0 "
                             "every speaker keeps all its clips up to the cap, a random subset above it")
    parser.add_argument("--seed", type=int, default=0, help="seed of the --random order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(
        f"{args.input.stem.rsplit('_', 1)[0]}_selected.csv")
    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    if not rows or (SCORE not in rows[0] and not args.random):
        raise SystemExit(f"{args.input} has no {SCORE} column")

    durations = np.array([duration(row["name"]) for row in rows])
    if args.random:
        scores, ranking = None, np.random.default_rng(args.seed).permutation(len(rows)).astype(float)
    else:
        scores = np.array([float(row[SCORE]) for row in rows])
        ranking = scores if args.raw else banded_z(scores, durations)
    budget = round(args.share * len(rows))

    print_test_title(f"Selecting from {DATASET}: {len(rows)} clips")
    print_info("input", args.input)
    print_info("output", output)
    print_info("budget", f"{budget} clips ({args.share:.0%})")
    print_info("bounds", f"{args.min_per_speaker}-{args.max_per_speaker} clips per speaker")
    print_info("ranking", f"random, seed {args.seed}" if args.random
               else f"raw {SCORE}" if args.raw else f"{SCORE} z within duration bands")

    chosen = select(rows, ranking, budget, args.min_per_speaker, args.max_per_speaker)
    write(output, [rows[index] for index in sorted(chosen)])
    report(rows, chosen, durations, scores, ranking, budget)


def duration(name: str) -> float:
    """Seconds, from the segmenter's name: <video>-<index>-<start cs>-<end cs>."""

    start, end = name.rsplit("-", 2)[1:]
    return (int(end) - int(start)) / 100


def banded_z(scores: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Each score as a z within its duration band; a band of one clip is left at 0."""

    z = np.zeros_like(scores)
    for low, high in BANDS:
        band = (durations >= low) & (durations < high)
        if band.sum() > 1:
            z[band] = (scores[band] - scores[band].mean()) / scores[band].std()
    return z


def select(rows: list, ranking: np.ndarray, budget: int, floor: int, cap: int) -> list:
    """Indices of the selection: every eligible speaker's best `floor`, then the best of the rest
    under `cap`, until `budget` is reached."""

    by_speaker = {}
    for index in np.argsort(-ranking, kind="stable"):
        by_speaker.setdefault(rows[index]["speaker_id"], []).append(int(index))
    eligible = {speaker: indices for speaker, indices in by_speaker.items() if len(indices) >= floor}

    chosen = [index for indices in eligible.values() for index in indices[:floor]]
    if len(chosen) > budget:
        raise SystemExit(f"The floors alone take {len(chosen)} clips ({len(eligible)} speakers x "
                         f"{floor}), over the budget of {budget}: raise --share or lower "
                         f"--min-per-speaker")

    # a speaker's clips past its cap never enter the candidates
    rest = sorted((index for indices in eligible.values() for index in indices[floor:cap]),
                  key=lambda index: -ranking[index])
    return chosen + rest[:budget - len(chosen)]


def write(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), **DIALECT)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list, chosen: list, durations: np.ndarray, scores: np.ndarray | None,
           ranking: np.ndarray, budget: int) -> None:
    speakers = {row["speaker_id"] for row in rows}
    per_speaker = {}
    for index in chosen:
        per_speaker[rows[index]["speaker_id"]] = per_speaker.get(rows[index]["speaker_id"], 0) + 1
    picked = np.zeros(len(rows), dtype=bool)
    picked[chosen] = True

    print_section("Selected")
    print_summary_line("  clips", f"{len(chosen)} of {len(rows)} (budget {budget})")
    print_info("total length", f"{durations[picked].sum() / 3600:.2f} h of "
                               f"{durations.sum() / 3600:.2f} h")
    print_info("speakers", f"{len(per_speaker)} of {len(speakers)}")
    print_info("clips per speaker", f"min {min(per_speaker.values())}   "
                                    f"max {max(per_speaker.values())}")
    print_info("mean length", f"{durations[picked].mean():.1f} s (pool {durations.mean():.1f} s)")
    if scores is not None:
        print_info(f"mean {SCORE}", f"{scores[picked].mean():.4f} (pool {scores.mean():.4f}, "
                                    f"rejected {scores[~picked].mean():.4f})")
        print_info("mean ranking score", f"{ranking[picked].mean():.3f} (pool {ranking.mean():.3f})")


if __name__ == "__main__":
    main()
