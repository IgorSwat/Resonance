"""Filter Emilia through the quality cascade and write the survivors to a CSV.

    python scripts/filter/emilia.py --batch en-b000000 --limit 500

Emilia ships one directory per tar shard (data/Emilia/en-b000000/), each clip an mp3 beside a
JSON sidecar holding its transcript, speaker and language. --batch restricts the run to a
single shard; without it every batch directory under --root is filtered in one pass, which is
rarely what you want — a shard is ~25k clips.

Clips that survive the cascade go through a second, source-level pass: Emilia's diarisation
leaks per source rather than per clip, so a speaker whose clips are rejected by the
multi-speaker stage often enough is dropped whole. Configure it with source_rejection_enabled
in the quality config.

Use --limit while calibrating: it draws a random sample, so the rejection breakdown it prints
tells you whether the configured bounds keep a sensible share of the corpus before committing
to a full pass.
"""

import argparse
import collections
import csv
import json
import multiprocessing
import pathlib
import random
import shutil
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_test_title
from scripts.filter.libritts import Reservoir, prepare, report
from tools.metrics.nisqa import FIELDS as NISQA_FIELDS, NisqaMetric
from tools.metrics.pipeline import Pipeline
from tools.metrics.types import QualityConfig, QualityVerdict

DATASET = "Emilia"
ROOT = pathlib.Path("data/Emilia")
CONFIG = pathlib.Path("configurations/quality_filtering_emilia.yaml")
REJECTED = pathlib.Path("tmp/rejected")
ACCEPTED = pathlib.Path("tmp/accepted")
DUMP_SAMPLE = 100                       # clips copied for listening, drawn uniformly
COLUMNS = ("dataset", "name", "transcription", "speaker_id", "language")
SEPARATOR = "|"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--batch", help="a single batch directory under --root, e.g. en-b000000")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("emilia_filtered.csv"))
    parser.add_argument("--limit", type=int, help="process N random files instead of the whole set")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--dump-rejected", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random clips rejected on quality (not length) "
                             f"to {REJECTED}/, with a report of the scores that failed")
    parser.add_argument("--dump-accepted", action="store_true",
                        help=f"copy {DUMP_SAMPLE} random accepted clips to {ACCEPTED}/, "
                             "with a report of their NISQA scores")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel processes; each loads its own copy of the models, so more "
                             "is not faster — 8 measured slower than 1 and ran out of memory")
    return parser.parse_args()


_pipeline = None


def _setup(config, verbose):
    """One pipeline per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline
    torch.set_num_threads(1)
    _pipeline = Pipeline(config, verbose=verbose)


def _score(path):
    try:
        audio, meta = load(path)
    except Exception as error:
        return path, None, str(error), 0.0, None
    verdict = _pipeline.run(audio, meta["text"])
    return path, verdict, None, len(audio["audio"]) / audio["sample_rate"], meta


def batches(root, batch):
    """The batch directories to filter: the one named, or every one under the root."""

    if batch:
        directory = root / batch
        if not directory.is_dir():
            raise SystemExit(f"No batch directory at {directory}")
        return [directory]
    return sorted(path for path in root.iterdir() if path.is_dir())


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()

    directories = batches(args.root, args.batch)
    sidecars = [path for directory in directories for path in sorted(directory.glob("*.json"))]
    if not sidecars:
        raise SystemExit(f"No Emilia clips under {', '.join(str(d) for d in directories)}")
    if args.limit:
        sidecars = random.Random(args.seed).sample(sidecars, min(args.limit, len(sidecars)))

    print_test_title(f"Filtering {DATASET}: {len(sidecars)} clips")
    print_info("root", args.root)
    print_info("batches", args.batch or f"all {len(directories)}")
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)

    order = [stage.verdict for stage in Pipeline(config).stages]
    second_pass = config.source_rejection_enabled and QualityVerdict.MULTI_SPEAKER in order
    print_info("source pass", f"reject a speaker above {config.source_max_flag_rate:.0%} flagged "
                              f"clips (min {config.source_min_clips})" if second_pass else "off")
    print_info("workers", args.workers)
    if args.dump_rejected:
        print_info("rejected sample", f"{DUMP_SAMPLE} clips -> {REJECTED}")
    if args.dump_accepted:
        print_info("accepted sample", f"{DUMP_SAMPLE} clips -> {ACCEPTED}")

    rejected_sample = Reservoir(DUMP_SAMPLE, args.seed)
    accepted_sample = Reservoir(DUMP_SAMPLE, args.seed)

    verdicts = collections.Counter()
    scored_clips, flagged_clips = collections.Counter(), collections.Counter()
    rows, accepted, failures, start = [], [], 0, time.time()
    for index, (path, verdict, error, duration, meta) in enumerate(
        scored(sidecars, config, args), 1
    ):
        if error:
            print(f"{Colors.FAIL}{path.stem}: {error}{Colors.ENDC}")
            failures += 1
            continue

        verdicts[verdict] += 1
        if reached(verdict, order):
            scored_clips[meta["speaker"]] += 1
            flagged_clips[meta["speaker"]] += verdict is QualityVerdict.MULTI_SPEAKER
        if verdict is QualityVerdict.ACCEPTED:
            rows.append((DATASET, path.stem, meta["text"], meta["speaker"], meta["language"]))
            accepted.append((meta["speaker"], duration))
            accepted_sample.add((path, duration, meta["speaker"]))
        elif verdict not in (QualityVerdict.TOO_SHORT, QualityVerdict.TOO_LONG):
            # length is a sourcing decision rather than a defect, so those are never dumped
            rejected_sample.add((path, verdict, duration))
        if index % 200 == 0:
            rate = index / (time.time() - start)
            print(f"  {index}/{len(sidecars)} clips, {rate:.1f}/s, "
                  f"{100 * len(rows) / index:.0f}% accepted", flush=True)

    if second_pass:
        sources = flagged_sources(scored_clips, flagged_clips, config)
        rows, accepted, dropped = drop_sources(sources, rows, accepted, accepted_sample)
        verdicts[QualityVerdict.ACCEPTED] -= dropped
        verdicts[QualityVerdict.MULTI_SPEAKER_SOURCE] += dropped
        print_info("sources rejected", f"{len(sources)} of {len(scored_clips)} speakers, "
                                       f"{dropped} of their clips")
    write(args.output, rows)

    if args.dump_rejected:
        dump_rejected(rejected_sample, config)
    if args.dump_accepted:
        dump_accepted(accepted_sample, config)
    report(verdicts, accepted, failures, time.time() - start, args.output)


def reached(verdict, order):
    """Whether the clip survived far enough for the multi-speaker stage to have scored it.

    The cascade stops at its first rejection, so without this a speaker whose clips mostly died
    on quality would look clean to the source pass simply for never having been scored.
    """

    if verdict is QualityVerdict.ACCEPTED:
        return True
    if verdict not in order:                    # rejected on length, before any stage ran
        return False
    return order.index(verdict) >= order.index(QualityVerdict.MULTI_SPEAKER)


def flagged_sources(scored_clips, flagged_clips, config):
    """Speakers the multi-speaker stage rejects often enough to distrust the whole source.

    An Emilia speaker is one voice in one recording, so the leak is a property of the source:
    pyannote merges clean turn-taking between similar voices, and a conversational source lands
    some of its clips in the accepted set however the stage is tuned. Over the labelled clips,
    the speaker's rate predicted a second voice better than the clip's own score did.
    """

    return {speaker for speaker, count in scored_clips.items()
            if count >= config.source_min_clips
            and flagged_clips[speaker] / count >= config.source_max_flag_rate}


def drop_sources(sources, rows, accepted, accepted_sample):
    """Remove every clip belonging to a rejected source, listening sample included."""

    kept = [row for row in rows if row[3] not in sources]
    accepted_sample.items = [item for item in accepted_sample.items if item[2] not in sources]
    return (kept, [item for item in accepted if item[0] not in sources], len(rows) - len(kept))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        # No Emilia transcript in this corpus contains SEPARATOR or a backslash, so the writer
        # never has to escape; quotechar=None leaves the ASR's own quotes as written.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def dump_rejected(sample, config):
    """Copy each sampled clip next to a note of what it scored and what was allowed.

    The scores are recomputed here, for the sample only — running describe() on every rejected
    clip would re-evaluate a metric the pipeline had already decided on.
    """

    prepare(REJECTED)
    pipeline = Pipeline(config)
    with open(REJECTED / "rejections.txt", "w") as report_file:
        for path, verdict, duration in sorted(sample.items, key=lambda item: item[1].value):
            audio, meta = load(path)
            shutil.copy2(path.with_suffix(".mp3"), REJECTED / f"{verdict.value}_{path.stem}.mp3")
            report_file.write(f"{path.stem}.mp3  ({duration:.2f} s)\n")
            report_file.write(f"  rejected by: {verdict.value}\n")
            report_file.write("\n".join(pipeline.describe(verdict, audio, meta["text"])) + "\n\n")
    print_info("rejected dumped", f"{len(sample.items)} of {sample.seen} to {REJECTED}")


def dump_accepted(sample, config):
    """Copy each sampled clip and record what NISQA thought of it."""

    prepare(ACCEPTED)
    nisqa = NisqaMetric(min_duration=config.min_duration, max_duration=config.nisqa_max_duration)
    with open(ACCEPTED / "nisqa.txt", "w") as report_file:
        for path, duration, _ in sorted(sample.items):
            audio, _ = load(path)
            shutil.copy2(path.with_suffix(".mp3"), ACCEPTED / f"{path.stem}.mp3")
            scores = nisqa.evaluate(audio)
            report_file.write(f"{path.stem}.mp3  ({duration:.2f} s)\n  ")
            report_file.write("  ".join(f"{field} {scores[field]:.2f}" for field in NISQA_FIELDS))
            report_file.write("\n\n")
    print_info("accepted dumped", f"{len(sample.items)} of {sample.seen} to {ACCEPTED}")


def scored(sidecars, config, args):
    """Verdicts for every clip, in a worker pool unless --workers 1."""

    if args.workers <= 1:
        _setup(config, args.verbose)
        yield from (_score(path) for path in sidecars)
        return
    with multiprocessing.Pool(args.workers, _setup, (config, args.verbose)) as pool:
        yield from pool.imap_unordered(_score, sidecars, chunksize=8)


def load(path):
    """The clip's audio and its JSON sidecar, given the path to the sidecar."""

    audio, sample_rate = sf.read(path.with_suffix(".mp3"), dtype="float32")
    return {"audio": audio, "sample_rate": sample_rate}, json.loads(path.read_text())


if __name__ == "__main__":
    main()
