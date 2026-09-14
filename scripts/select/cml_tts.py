"""Select a prosodically varied subset of the filtered CML-TTS CSV.

    python scripts/select/cml_tts.py --lang ES --min-per-speaker 3 --target-per-speaker 7

Takes the output of scripts/filter/cml_tts.py as the input pool, scores every clip with
tools/prosody, and picks clips whose prosody-cell histogram is as flat as the pool allows. Every
speaker holding --min-per-speaker clips is admitted and contributes --target-per-speaker of them,
or all they have if fewer, then grows toward --max-per-speaker for as long as each further clip
still flattens the histogram, so the subset size is decided by the pool rather than fixed up
front. A speaker here is a LibriVox narrator — a real person with a deep pool of clips, unlike
Emilia's per-source diarisation labels.

Male voices outnumber female ones in the LibriVox corpora, so speakers above PITCH_EDGE get their
own --high-target-per-speaker / --high-max-per-speaker instead, contributing more clips each. The
edge is measured from the audio: CML-TTS ships no gender column.

The audio lives in the parquet shards rather than on disk, so each prosody pass is a scan of
--root/<language> instead of a seek per clip; see score().
"""

import argparse
import collections
import csv
import pathlib
import random
import sys
import time

import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.fetch.cml_tts import LANGUAGES
from scripts.filter.cml_tts import DATASET, ROOT, SEPARATOR, clip_name, decode, rows
from scripts.select.libritts import capped_random, report
from tools.prosody.constants import (DEFAULT_CEILING, DEFAULT_FLOOR, HIGH_TARGET, MIN_DURATION,
                                     PITCH_EDGE)
from tools.prosody.contour import f0_contour
from tools.prosody.features import prosody_features
from tools.prosody.normalize import (adapted_range, clip_median, speaker_reference, to_semitones,
                                     within_octave_guard)
from tools.prosody.selection import select_bounded

ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
REPORTED = ("st_range", "st_slope_std", "st_final_slope", "voiced_onsets_per_s",
            "voiced_frac", "n_phrases")
PROGRESS_EVERY = 500


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="PL", type=str.upper, choices=sorted(LANGUAGES))
    parser.add_argument("--input", type=pathlib.Path,
                        default=pathlib.Path("cml_tts_filtered.csv"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("cml_tts_selected.csv"))
    parser.add_argument("--root", type=pathlib.Path, default=ROOT,
                        help="the parquet shards the pool's audio is read from")
    parser.add_argument("--min-per-speaker", type=int, default=DEFAULT_FLOOR,
                        help="clips a speaker must have to be kept at all")
    parser.add_argument("--target-per-speaker", type=int, default=DEFAULT_FLOOR,
                        help="clips a kept speaker contributes, or all they have if fewer")
    parser.add_argument("--max-per-speaker", type=int, default=DEFAULT_CEILING,
                        help="most a speaker may contribute, reached only by flattening the histogram")
    parser.add_argument("--high-target-per-speaker", type=int, default=HIGH_TARGET,
                        help=f"clips a speaker above {PITCH_EDGE:.0f} Hz contributes, "
                             "or all they have if fewer")
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
    directory = args.root / LANGUAGES[args.lang]
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No {DATASET} shards under {directory}")

    print_test_title(f"Selecting from {DATASET} {args.lang}: {len(pool)} filtered clips")
    print_info("input", args.input)
    print_info("output", args.output)
    print_info("shards", f"{len(shards)} under {directory}")
    print_info("bounds", f"{args.target_per_speaker}-{args.max_per_speaker} clips per speaker  "
                         f"({args.high_target_per_speaker}-{args.high_max_per_speaker} "
                         f"above {PITCH_EDGE:.0f} Hz)")
    print_info("admitted", f"speakers with at least {args.min_per_speaker} clips")

    start = time.time()
    scored, rejected = score(pool, shards)
    print_info("scored", f"{len(scored)} clips in {time.time() - start:.0f}s"
                         f"  (rejected {dict(rejected)})")
    if not scored:
        raise SystemExit("Nothing to select from")

    selected = select_bounded(scored, args.min_per_speaker, args.max_per_speaker, seed=args.seed,
                              target=args.target_per_speaker,
                              high_bounds=(args.high_target_per_speaker, args.high_max_per_speaker))
    if not selected:
        raise SystemExit(f"No speaker has {args.min_per_speaker} usable clips")
    print_info("high-voiced", f"{high_share(scored):.1%} of the pool "
                              f"-> {high_share(selected):.1%} of the selection")
    write(args.output, selected, languages)

    baseline = capped_random(scored, len(selected), args.max_per_speaker, args.seed)
    report(scored, selected, baseline, args.output)
    if args.dump_accepted:
        dump_accepted(selected, shards, args.seed)


def score(pool, shards):
    """Two passes per the prosody recipe: a pitch reference per speaker, then features.

    scripts/select/libritts.py opens each clip twice by path, once for its pitch median and once
    for a contour tracked inside its speaker's adapted range. A parquet row has no path and the
    second pass cannot reuse the first one's contour, so each pass is a scan of the shards
    instead of a seek per clip. Memory stays flat: only a float per clip survives pass one.
    """

    wanted = {name: (transcript, speaker) for name, transcript, speaker in pool}
    rejected = collections.Counter()

    medians, found = {}, 0
    for name, audio in extract(shards, set(wanted)):
        found += 1
        if len(audio["audio"]) / audio["sample_rate"] < MIN_DURATION:
            rejected["too short"] += 1
        elif (median := clip_median(f0_contour(audio))) is None:
            rejected["unvoiced"] += 1
        else:
            medians[name] = median
        if found % PROGRESS_EVERY == 0:
            print(f"  pass 1: {found}/{len(wanted)} clips", flush=True)
    if found < len(wanted):
        rejected["not in shards"] = len(wanted) - found

    by_speaker = collections.defaultdict(list)
    for name in medians:
        by_speaker[wanted[name][1]].append(name)
    references = {speaker: speaker_reference([medians[name] for name in names])
                  for speaker, names in by_speaker.items()}

    keep = set()
    for name, median in medians.items():
        if within_octave_guard(median, references[wanted[name][1]]):
            keep.add(name)
        else:
            rejected["octave guard"] += 1

    scored = []
    for index, (name, audio) in enumerate(extract(shards, keep), 1):
        transcript, speaker = wanted[name]
        reference = references[speaker]
        duration = len(audio["audio"]) / audio["sample_rate"]
        contour = to_semitones(f0_contour(audio, *adapted_range(reference)), reference)
        features = prosody_features(contour, duration)
        if features is None:
            rejected["no features"] += 1
            continue
        scored.append(features | {"name": name, "transcript": transcript, "speaker": speaker,
                                  "ref_hz": reference, "dur": duration})
        if index % PROGRESS_EVERY == 0:
            print(f"  pass 2: {index}/{len(keep)} clips", flush=True)
    return scored, rejected


def extract(shards, names):
    """Each wanted clip's audio, in one scan of the shards, stopping once they are all found."""

    remaining = set(names)
    for row in rows(shards):
        name = clip_name(row)
        if name in remaining:
            remaining.discard(name)
            yield name, decode(row)
            if not remaining:
                return


def high_share(scored):
    """Fraction of clips spoken above PITCH_EDGE — the imbalance the high bounds correct."""

    return sum(row["ref_hz"] > PITCH_EDGE for row in scored) / len(scored)


def read_pool(path):
    """The filtered CSV, as (name, transcript, speaker) triples and a name -> language map."""

    if not path.exists():
        raise SystemExit(f"No input pool at {path}; run scripts/filter/cml_tts.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    return ([(row["name"], row["transcription"], row["speaker_id"]) for row in rows],
            {row["name"]: row["language"] for row in rows})


def write(path, selected, languages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, **DIALECT)
        writer.writerow(COLUMNS)
        for row in sorted(selected, key=lambda row: row["name"]):
            writer.writerow((DATASET, row["name"], row["transcript"], row["speaker"],
                             languages[row["name"]]))


def dump_accepted(selected, shards, seed):
    """Copy a listening sample next to what each clip measured."""

    ACCEPTED.mkdir(parents=True, exist_ok=True)
    for stale in ACCEPTED.iterdir():
        if stale.is_file():
            stale.unlink()

    sample = random.Random(seed).sample(selected, min(DUMP_SAMPLE, len(selected)))
    audios = dict(extract(shards, {row["name"] for row in sample}))
    with open(ACCEPTED / "prosody.txt", "w") as report_file:
        for row in sorted(sample, key=lambda row: row["st_range"]):
            audio = audios[row["name"]]
            sf.write(ACCEPTED / f"range{row['st_range']:05.2f}_{row['name']}.wav",
                     audio["audio"], audio["sample_rate"])
            report_file.write(f"{row['name']}.wav  ({row['dur']:.2f} s, "
                              f"ref {row['ref_hz']:.0f} Hz, cell {row['cell']})\n  ")
            report_file.write("  ".join(f"{field} {row[field]:.2f}" for field in REPORTED))
            report_file.write("\n\n")
    print_info("dumped", f"{len(sample)} clips to {ACCEPTED}")


if __name__ == "__main__":
    main()
