"""Merge the Emilia speaker IDs that are one person recorded in several sources.

    python scripts/filter/emilia_speaker_fixup.py \
        --input data/processed/emilia_en-b000000_filtered.csv \
        --output data/processed/emilia_en-b000000_speakers.csv

Emilia diarises each source on its own and never links speakers across sources, so one person in
two recordings gets two unrelated IDs (knowledge/emilia.md §1). That inflates the speaker count,
distorts the per-speaker prosody reference in scripts/select/emilia.py, and leaks a voice across
a train/validation split.

Reads a CSV from scripts/filter/emilia.py, embeds a few clips per source with
pyannote/wespeaker-voxceleb-resnet34-LM, clusters the sources the model calls one person, and
rewrites speaker_id to the lowest ID of each cluster. No other column changes.

Sources are compared over their clip pairs rather than as one averaged vector, because a
centroid's scale moves with the clip count: over 15 unrelated sources the cosine ceiling climbs
0.735 (1 clip) -> 0.869 (4) -> 0.917 (16) and overtakes true matches from 3 clips up, so a
threshold calibrated on one run size is wrong on the next. The clip-pair fraction holds still —
60-64 merges whether 1 or 4 clips are sampled.
"""

import argparse
import collections
import csv
import itertools
import pathlib
import sys
import time

import librosa
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import (Colors, print_info, print_section, print_separator,
                               print_summary_line, print_test_title)

ROOT = pathlib.Path("data/Emilia")
EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
SEPARATOR = "|"
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
SAMPLE_RATE = 16000
CLIP_SECONDS = 8.0              # the model saturates well before this
CLIP_PAIR = 0.75                # cosine at which one clip of A and one of B count as agreeing
DUPLICATE = 0.95                # a clip pair this close is the same recording, not a speaker link


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, required=True,
                        help="a CSV from scripts/filter/emilia.py")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
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

    print_test_title(f"Speaker fixup: {len(rows)} clips, {len(clips)} source IDs")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("threshold", f">= {args.threshold:.0%} of clip pairs above cosine {CLIP_PAIR}, "
                            f"{args.linkage} linkage")

    paths = locate(args.root, {row["name"] for row in rows})
    unlocatable = sum(1 for names in clips.values() if not any(n in paths for n in names))
    if unlocatable:
        print(f"{Colors.WARNING}  {unlocatable} sources have no audio under {args.root}; "
              f"left unmerged{Colors.ENDC}")

    start = time.time()
    embeddings = speaker_embeddings(clips, paths, args)
    print_info("embedded", f"{len(embeddings)} sources, "
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
        raise SystemExit(f"No input at {path}; run scripts/filter/emilia.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    if rows and "speaker_id" not in rows[0]:
        raise SystemExit(f"{path} has no speaker_id column: {list(rows[0])}")
    return rows


def locate(root, names):
    """Map clip name to its mp3, walking the batch directories once rather than per clip."""

    return {path.stem: path for path in root.rglob("*.mp3") if path.stem in names}


def speaker_embeddings(clips, paths, args):
    """The per-clip vectors of each source, L2-normalised, longest clips first.

    Kept per clip rather than averaged: a centroid blurs a source holding more than one voice.
    """

    model = embedder(args.device)
    embeddings = {}
    for index, (speaker, names) in enumerate(sorted(clips.items()), 1):
        vectors = []
        for name in longest(names, paths, args.clips_per_speaker):
            audio, _ = librosa.load(paths[name], sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
            with torch.no_grad():
                vector = model(torch.tensor(audio, dtype=torch.float32)[None, None])[0]
            vectors.append(vector / np.linalg.norm(vector))
        if vectors:
            embeddings[speaker] = np.stack(vectors)
        if index % 100 == 0:
            print(f"  {index}/{len(clips)} sources", flush=True)
    return embeddings


def longest(names, paths, count):
    """The longest clips of one source, by file size — a proxy that needs no decode."""

    known = [name for name in names if name in paths]
    return sorted(known, key=lambda name: -paths[name].stat().st_size)[:count]


def embedder(device=None):
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    return PretrainedSpeakerEmbedding(EMBEDDING, device=torch.device(device))


def compare(embeddings):
    """Per source pair: the share of clip pairs that agree, and their closest clip pair.

    The share drives the clustering; the closest pair only tells the report whether a merge is
    duplicated audio rather than one person recorded twice.
    """

    speakers = sorted(embeddings)
    agreement, closest = np.eye(len(speakers)), np.ones((len(speakers),) * 2)
    for i, j in itertools.combinations(range(len(speakers)), 2):
        cosines = embeddings[speakers[i]] @ embeddings[speakers[j]].T
        agreement[i, j] = agreement[j, i] = float((cosines > CLIP_PAIR).mean())
        closest[i, j] = closest[j, i] = float(cosines.max())
    return agreement, closest


def merge_map(speakers, agreement, threshold, method="single"):
    """Source ID -> the lowest ID of its cluster, for the ones that move.

    Single linkage because the errors are not symmetric: merging two similar voices costs a
    little diversity, splitting one person across IDs costs prosody references and split
    integrity. Complete linkage splits a person on their weakest pair alone.
    """

    if len(speakers) < 2:
        return {}

    distance = np.clip(1 - agreement, 0, None)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method=method),
                      1 - threshold, criterion="distance")

    clusters = collections.defaultdict(list)
    for speaker, label in zip(speakers, labels):
        clusters[label].append(speaker)
    return {speaker: min(members) for members in clusters.values() if len(members) > 1
            for speaker in members if speaker != min(members)}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), **DIALECT)
        writer.writeheader()
        writer.writerows(rows)


def report(clips, speakers, agreement, closest, merges, args):
    """What merged into what, how strongly, and whether it is duplicated audio."""

    index = {speaker: position for position, speaker in enumerate(speakers)}
    families = collections.defaultdict(list)
    for speaker, canonical in merges.items():
        families[canonical].append(speaker)

    print_section("Merged sources")
    if not merges:
        print("  none")
    for canonical, members in sorted(families.items(), key=lambda item: -len(item[1])):
        group = sorted(members + [canonical])
        pairs = [(index[a], index[b]) for i, a in enumerate(group) for b in group[i + 1:]]
        agree = [agreement[i, j] for i, j in pairs]
        nearest = max(closest[i, j] for i, j in pairs)
        print(f"  {canonical[-6:]} <- {', '.join(m[-6:] for m in sorted(members))}"
              f"   agreement {min(agree):.0%}-{max(agree):.0%}"
              f"   {sum(len(clips[m]) for m in group)} clips")
        if nearest >= DUPLICATE:
            print(f"    {Colors.WARNING}closest clip pair {nearest:.2f}: duplicated audio, "
                  f"not a speaker link{Colors.ENDC}")

    print_section("Result")
    print_summary_line("  sources", f"{len(clips)} -> {len(clips) - len(merges)} "
                                    f"({Colors.OKGREEN}{len(merges)} merged away{Colors.ENDC})")
    print_info("clips relabelled", sum(len(clips[speaker]) for speaker in merges))
    print_info("largest cluster", max((len(m) + 1 for m in families.values()), default=1))
    print_info("unembedded sources", len(clips) - len(speakers))
    print_separator()
    print_info("written to", args.output)


if __name__ == "__main__":
    main()
