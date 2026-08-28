"""Remove multi-speaker clips from a finalized Emilia subset.

    python scripts/finalize/emilia_prune.py --root data/processed/1
    python scripts/finalize/emilia_prune.py --root data/processed/1 --multispeakers list.txt

Takes a subset as scripts/finalize/emilia.py leaves it — an audio/ directory, a codecs/ directory
and the CSVs beside them — and deletes every flagged clip from all three: the mp3, the .npy, and
its row in every CSV under --root.

Without --multispeakers the subset is scanned with SpeakerDriftMetric and the flagged names are
written to multi_speaker_candidates.txt under --root before anything is removed; with it, that
scan is skipped and the file's names are used as given. Deletion is irreversible, so --dry-run
reports what would go without touching the subset.

Flagging is by duration-normalized z rather than the metric's own bound: raw similarity falls
with clip length (r about -0.55 on batches 0 and 7), so a flat threshold is largely a long-clip
filter. See knowledge/emilia.md §3.
"""

import argparse
import csv
import pathlib
import sys
import warnings

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from tools.metrics.speaker_drift import SpeakerDriftMetric

CANDIDATES = "multi_speaker_candidates.txt"
SEPARATOR = "|"
NAME_COLUMN = "name"
# Wide enough that every band holds hundreds of clips in a 10k subset, narrow enough that the
# median similarity is flat inside one.
BANDS = ((0, 6), (6, 9), (9, 13), (13, 18), (18, float("inf")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="subset directory holding audio/, codecs/ and the CSVs")
    parser.add_argument("--multispeakers", type=pathlib.Path,
                        help=f"clip names to remove, one per line; default is to scan and write "
                             f"{CANDIDATES} under --root")
    parser.add_argument("--max-z", type=float, default=-2.0,
                        help="flag clips at or below this z within their duration band")
    parser.add_argument("--device", help="torch device for ECAPA; MPS is supported")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be removed, then stop")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir, codec_dir = args.root / "audio", args.root / "codecs"
    if not audio_dir.is_dir():
        raise SystemExit(f"No audio directory under {args.root}")

    print_test_title(f"Pruning {args.root}")
    if args.multispeakers:
        names = read_names(args.multispeakers)
        print_info("flagged", f"{len(names)} clips from {args.multispeakers}")
    else:
        names = scan(audio_dir, args.max_z, args.device)
        listing = args.root / CANDIDATES
        listing.write_text("".join(f"{name}\n" for name in names))
        print_info("flagged", f"{len(names)} clips (z <= {args.max_z}) -> {listing}")
    if not names:
        raise SystemExit("Nothing flagged; subset unchanged")

    csvs = sorted(args.root.glob("*.csv"))
    print_section("Removing")
    print_info("audio", f"{count(audio_dir, names, '.mp3')} of {len(names)} mp3s present")
    print_info("codecs", f"{count(codec_dir, names, '.npy')} of {len(names)} npys present")
    for path in csvs:
        print_info(path.name, f"{rows_matching(path, names)} rows")
    if args.dry_run:
        print(f"\n{Colors.WARNING}--dry-run: nothing removed{Colors.ENDC}")
        return

    print_section("Removed")
    print_info("audio", delete(audio_dir, names, ".mp3"))
    print_info("codecs", delete(codec_dir, names, ".npy"))
    for path in csvs:
        print_info(path.name, f"{prune_csv(path, names)} rows")


def read_names(path):
    """
    Clip names from a listing, one per line, with any extension dropped.
    """

    return {line.strip().removesuffix(".mp3") for line in path.read_text().splitlines()
            if line.strip()}


def scan(audio_dir, max_z, device):
    """
    Names whose ECAPA similarity is at or below max_z within their duration band.
    """

    metric = SpeakerDriftMetric(device=device)
    paths = sorted(audio_dir.glob("*.mp3"))
    names, durations, scores = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, path in enumerate(paths, 1):
            y, sample_rate = sf.read(path, dtype="float32")
            names.append(path.stem)
            durations.append(len(y) / sample_rate)
            scores.append(metric.evaluate({"audio": y, "sample_rate": sample_rate})["similarity"])
            if index % 1000 == 0:
                print(f"  scanned {index}/{len(paths)}", flush=True)

    durations, scores = np.array(durations), np.array(scores)
    z = np.zeros_like(scores)
    for low, high in BANDS:
        band = (durations >= low) & (durations < high)
        # A band with one clip has no spread to normalize by, and is left at z = 0.
        if band.sum() > 1:
            z[band] = (scores[band] - scores[band].mean()) / scores[band].std()
    return [names[i] for i in np.argsort(z) if z[i] <= max_z]


def count(directory, names, suffix):
    return sum((directory / f"{name}{suffix}").exists() for name in names) if directory.is_dir() else 0


def delete(directory, names, suffix):
    if not directory.is_dir():
        return 0
    removed = 0
    for name in names:
        path = directory / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def rows_matching(path, names):
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter=SEPARATOR, quotechar=None, quoting=csv.QUOTE_NONE)
        column = next(reader).index(NAME_COLUMN)
        return sum(row[column].removesuffix(".mp3") in names for row in reader if row)


def prune_csv(path, names):
    """
    Drop the flagged rows, rewriting the file only once every row has been read.
    """

    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter=SEPARATOR, quotechar=None, quoting=csv.QUOTE_NONE)
        header = next(reader)
        column = header.index(NAME_COLUMN)
        rows = [row for row in reader if row]
    kept = [row for row in rows if row[column].removesuffix(".mp3") not in names]

    with open(path, "w", newline="") as handle:
        # The writer settings scripts/filter/emilia.py wrote these files with.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(header)
        writer.writerows(kept)
    return len(rows) - len(kept)


if __name__ == "__main__":
    main()
