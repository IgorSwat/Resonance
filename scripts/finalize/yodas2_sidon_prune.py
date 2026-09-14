"""Remove multi-speaker clips from a finalized YODAS2-Sidon subset.

    python scripts/finalize/yodas2_sidon_prune.py --root data/processed/YODAS2-Sidon/subset
    python scripts/finalize/yodas2_sidon_prune.py --root <subset> --multispeakers list.txt

Takes a subset as scripts/finalize/yodas2_sidon.py leaves it — an audio/ directory, a codecs/
directory and the CSVs beside them — and deletes every flagged clip from all three: the wav, the
.npy, and its row in every CSV under --root.

This is the third and last guard against a second voice, and it catches what the other two
cannot. The cascade's MultiSpeakerMetric asks who speaks when inside a window, so it merges two
similar voices taking clean turns; the filter's source pass judges a whole recording by its flag
rate, so it cannot save a good recording's one bad clip. SpeakerDriftMetric here asks only
whether the voice itself changes between two moments of the clip, which is the case those two
leave behind.

Without --multispeakers the subset is scanned and the flagged names are written to
multi_speaker_candidates.txt under --root before anything is removed; with it, that scan is
skipped and the file's names are used as given. Deletion is irreversible, so --dry-run reports
what would go without touching the subset.

Flagging is by duration-normalized z rather than the metric's own bound: raw similarity falls
with clip length, so a flat threshold is largely a long-clip filter. See knowledge/emilia.md §3.
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.finalize.emilia_prune import BANDS, count, delete, prune_csv, rows_matching
from tools.metrics.speaker_drift import SpeakerDriftMetric

CANDIDATES = "multi_speaker_candidates.txt"
SUFFIX = ".wav"


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
    print_info("audio", f"{count(audio_dir, names, SUFFIX)} of {len(names)} wavs present")
    print_info("codecs", f"{count(codec_dir, names, '.npy')} of {len(names)} npys present")
    for path in csvs:
        print_info(path.name, f"{rows_matching(path, names)} rows")
    if args.dry_run:
        print(f"\n{Colors.WARNING}--dry-run: nothing removed{Colors.ENDC}")
        return

    print_section("Removed")
    print_info("audio", delete(audio_dir, names, SUFFIX))
    print_info("codecs", delete(codec_dir, names, ".npy"))
    for path in csvs:
        print_info(path.name, f"{prune_csv(path, names)} rows")


def read_names(path):
    """Clip names from a listing, one per line, with any extension dropped."""

    return {line.strip().removesuffix(SUFFIX) for line in path.read_text().splitlines()
            if line.strip()}


def scan(audio_dir, max_z, device):
    """Names whose ECAPA similarity is at or below max_z within their duration band."""

    metric = SpeakerDriftMetric(device=device)
    paths = sorted(audio_dir.glob(f"*{SUFFIX}"))
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


if __name__ == "__main__":
    main()
