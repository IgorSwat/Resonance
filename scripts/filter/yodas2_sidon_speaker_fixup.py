"""Merge the YODAS2-Sidon source IDs that are one person recorded in several videos.

    python scripts/filter/yodas2_sidon_speaker_fixup.py \
        --input yodas2_sidon_filtered.csv \
        --output yodas2_sidon_speakers.csv

The corpus carries no speaker field at all, so scripts/filter/yodas2_sidon.py writes the video ID
into speaker_id and that is the only grouping available. A channel host appears in every video
they publish, so the same person routinely holds dozens of unrelated IDs — the same defect as
Emilia's per-source diarisation labels (knowledge/emilia.md §1), and worse here, since a YouTube
channel is a far stronger reason to expect repeats than two audiobooks are.

Reads a CSV from scripts/filter/yodas2_sidon.py, embeds a few clips per video with
pyannote/wespeaker-voxceleb-resnet34-LM, clusters the videos the model calls one person, and
rewrites speaker_id to the lowest ID of each cluster. No other column changes.

The method is Emilia's and this script is mostly its code: sources are compared over their clip
pairs rather than as one averaged vector, because a centroid's scale moves with the clip count
and a threshold calibrated on one run size is wrong on the next. Comparing per clip also survives
a video that holds more than one voice, which is the ordinary case here rather than the exception
— run this after the filter's source pass, which drops the worst of them.
"""

import argparse
import collections
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.filter.emilia_speaker_fixup import (DIALECT, compare, merge_map, report,
                                                 speaker_embeddings, write)
from scripts.filter.yodas2_sidon import AUDIO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, required=True,
                        help="a CSV from scripts/filter/yodas2_sidon.py")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=AUDIO,
                        help="where the filter wrote its clips")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="fraction of clip pairs that must agree for one person")
    parser.add_argument("--linkage", default="single",
                        choices=("average", "complete", "single", "weighted"))
    parser.add_argument("--clips-per-speaker", type=int, default=4,
                        help="clips embedded per source; every pair of them is compared")
    parser.add_argument("--device", help="torch device for the embedder")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.input)
    clips = collections.defaultdict(list)
    for row in rows:
        clips[row["speaker_id"]].append(row["name"])

    print_test_title(f"Speaker fixup: {len(rows)} clips, {len(clips)} video IDs")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("threshold", f">= {args.threshold:.0%} of clip pairs, {args.linkage} linkage")

    paths = locate(args.root, {row["name"] for row in rows})
    print_info("audio", f"{len(paths)} clips under {args.root}")
    unlocatable = sum(1 for names in clips.values() if not any(n in paths for n in names))
    if unlocatable:
        print(f"{Colors.WARNING}  {unlocatable} videos have no audio under {args.root}; "
              f"left unmerged{Colors.ENDC}")

    start = time.time()
    embeddings = speaker_embeddings(clips, paths, args)
    print_info("embedded", f"{len(embeddings)} videos, "
                           f"{sum(len(v) for v in embeddings.values())} clips "
                           f"in {time.time() - start:.0f}s")

    agreement, closest = compare(embeddings)
    merges = merge_map(sorted(embeddings), agreement, args.threshold, args.linkage)
    for row in rows:
        row["speaker_id"] = merges.get(row["speaker_id"], row["speaker_id"])
    write(args.output, rows)
    report(clips, sorted(embeddings), agreement, closest, merges, args)


def read_rows(path):
    if not path.exists():
        raise SystemExit(f"No input at {path}; run scripts/filter/yodas2_sidon.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    if rows and "speaker_id" not in rows[0]:
        raise SystemExit(f"{path} has no speaker_id column: {list(rows[0])}")
    return rows


def locate(root, names):
    """Map clip name to its wav, walking --root once rather than per clip."""

    return {path.stem: path for path in root.rglob("*.wav") if path.stem in names}


if __name__ == "__main__":
    main()
