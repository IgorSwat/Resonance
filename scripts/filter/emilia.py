"""Filter Emilia through two phases and write the survivors to a CSV.

    python scripts/filter/emilia.py --batch en-b000000 --limit 500

Emilia ships one directory per tar shard (data/Emilia/en-b000000/), each clip an mp3 beside a
JSON sidecar holding its transcript, speaker and language. --batch restricts the run to a single
shard; without it every batch under --root is filtered as one pool, which is what the phase-1
rules want — a source duplicated across two shards is only visible when both are in scope.

**Phase 1, preprocessing** works from the sidecars alone, no audio decoded:

- whole sources whose transcripts duplicate an earlier source are dropped (Emilia re-releases
  the same recording under several speaker IDs; see knowledge/emilia.md §6);
- numbers are spelled out as words in the clip's own language. The rewrite is per clip and its
  year/cardinal decision needs the audio, so it runs as each clip is read rather than up front.

**Phase 2, filtering** is the metric cascade of tools/metrics/pipeline.py, followed by a
source-level pass: Emilia's diarisation leaks per source rather than per clip, so a speaker the
multi-speaker stage rejects often enough is dropped whole.

Use --limit while calibrating: it draws a random sample, so the rejection breakdown it prints
tells you whether the configured bounds keep a sensible share of the corpus before committing to
a full pass. Phase 1 always sees the whole batch, since both of its rules judge a source by all
of its clips.
"""

import argparse
import collections
import csv
import dataclasses
import itertools
import json
import multiprocessing
import pathlib
import random
import re
import shutil
import sys
import time

import soundfile as sf
import torch
from num2words import num2words

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
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

# Thousands separators are read the English way, which is how the ASR wrote them; a lone comma
# between digits is left alone rather than guessed at.
NUMBER = re.compile(r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
                    r"(?P<ordinal>st|nd|rd|th)?\b", re.IGNORECASE)
YEAR_RANGE = range(1000, 2501)          # numbers a speaker may read either as a year or a count


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


def main():
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    sidecars = discover(args)
    shards = sorted({path.parent.name for path in sidecars})

    print_test_title(f"Filtering {DATASET}: {len(sidecars)} clips")
    print_info("root", args.root)
    print_info("batches", args.batch or
               f"all {len(shards)}: {', '.join(shards)}" if not args.batch else args.batch)
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("workers", args.workers)

    print_section("Phase 1 — preprocessing")
    sidecars = preprocess(sidecars, config, args)

    print_section("Phase 2 — filtering")
    start = time.time()
    outcome = filter_clips(sidecars, config, args)
    write(args.output, outcome.rows)

    if args.dump_rejected:
        dump_rejected(outcome.rejected_sample, config)
    if args.dump_accepted:
        dump_accepted(outcome.accepted_sample, config)
    report(outcome.verdicts, outcome.accepted, outcome.failures, time.time() - start, args.output)


def discover(args):
    """Every clip's sidecar, from the named batch directory or from all of them at once.

    Speaker IDs are unique across shards — a batch's speakers are split over ten tars by index
    modulo ten — so pooling shards needs no renaming and lets phase 1 see cross-shard duplicates.
    """

    if args.batch:
        directories = [args.root / args.batch]
        if not directories[0].is_dir():
            raise SystemExit(f"No batch directory at {directories[0]}")
    else:
        directories = sorted(path for path in args.root.iterdir() if path.is_dir())

    sidecars = [path for directory in directories for path in sorted(directory.glob("*.json"))]
    if not sidecars:
        raise SystemExit(f"No Emilia clips under {', '.join(str(d) for d in directories)}")
    return sidecars


# ---------------------------------------------------------------------------------------------
# Phase 1 — preprocessing: source deduplication, then per-clip transcript normalization
# ---------------------------------------------------------------------------------------------


def preprocess(sidecars, config, args):
    """The clips worth scoring: every shard in scope, minus every duplicated source.

    --limit is applied here, after deduplication, so a sampled run still drops the same sources a
    full run would: both phase-1 rules judge a source by all of its clips, not by the sample.
    """

    if config.duplicate_sources_enabled:
        duplicates = duplicate_sources(sidecars, config)
        kept = [path for path in sidecars if source_of(path) not in duplicates]
        print_info("duplicate sources", f"{len(duplicates)} sources dropped, "
                                        f"{len(sidecars) - len(kept)} clips")
        sidecars = kept
    else:
        print_info("duplicate sources", "off")

    if args.limit:
        sidecars = random.Random(args.seed).sample(sidecars, min(args.limit, len(sidecars)))
        print_info("limit", f"{len(sidecars)} clips sampled")

    print_info("verbalize numbers", verbalize_status(sidecars, config)
               if config.verbalize_numbers_enabled else "off")
    return sidecars


def duplicate_sources(sidecars, config):
    """Sources that re-release an earlier source's recording, by transcript overlap.

    Emilia gives every source its own speaker ID even when the audio is a re-encode of another
    source's, so a podcast's fixed intro turns into dozens of near-identical IDs. Two sources
    count as the same recording when over duplicate_max_overlap of the smaller one's transcripts
    also appear in the other.

    Of such a pair the larger source is kept, ties broken towards the lower ID. Keeping the
    smaller would throw away far more than the duplication: what repeats between two episodes of
    one podcast is its intro and outro, so the source holding only those is exactly the one worth
    losing.

    Only pairs that share at least one transcript are compared, so this stays linear in practice
    rather than quadratic in the number of sources.
    """

    lines, clips = collections.defaultdict(set), collections.Counter()
    for path in sidecars:
        meta = json.loads(path.read_text())
        clips[meta["speaker"]] += 1
        if line := normalize_text(meta["text"]):
            lines[meta["speaker"]].add(line)

    sources_of_line = collections.defaultdict(set)
    for source, spoken in lines.items():
        for line in spoken:
            sources_of_line[line].add(source)

    pairs = {pair for sources in sources_of_line.values() if len(sources) > 1
             for pair in itertools.combinations(sorted(sources), 2)}
    duplicates = set()
    for low, high in pairs:
        if (len(lines[low] & lines[high])
                > config.duplicate_max_overlap * min(len(lines[low]), len(lines[high]))):
            duplicates.add(high if clips[high] <= clips[low] else low)
    return duplicates


def normalize_text(text):
    """Transcripts compared for sameness: case, punctuation and spacing are not evidence."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def verbalize(text, language, audio=None):
    """Digits spelled out as words, in the language the sidecar declares.

    A number in YEAR_RANGE has two plausible readings — 1999 as "nineteen ninety-nine" or as "one
    thousand, nine hundred and ninety-nine" — and only the audio settles which was spoken, so
    each is rendered both ways and scored against the aligner over one shared emission. Numbers
    are decided independently: n of them cost n + 1 candidates, not 2^n.

    Falls back to the cardinal reading without an aligner or audio, and leaves digits alone in a
    language num2words cannot spell. Not to be trusted for Japanese, whose year form is an era
    name (1999 -> 平成十一年) rather than a reading.
    """

    cardinal = render(text, language)
    ambiguous = [index for index, match in enumerate(NUMBER.finditer(text))
                 if not match.group("ordinal") and match.group("number").isdigit()
                 and int(match.group("number")) in YEAR_RANGE]
    if not ambiguous or audio is None or _aligner is None:
        return cardinal

    candidates = [cardinal] + [render(text, language, {index}) for index in ambiguous]
    scores = _aligner.losses(audio, candidates)
    as_years = {index for index, score in zip(ambiguous, scores[1:]) if score < scores[0]}
    return render(text, language, as_years) if as_years else cardinal


def render(text, language, as_years=()):
    """The transcript with every number spelled out, reading the ones in as_years as years."""

    position = itertools.count()

    def spell(match):
        index = next(position)
        digits = match.group("number").replace(",", "")
        value = float(digits) if "." in digits else int(digits)
        if match.group("ordinal") and isinstance(value, int):
            kind = "ordinal"
        else:
            kind = "year" if index in as_years else "cardinal"
        try:
            return num2words(value, lang=language, to=kind)
        except (NotImplementedError, OverflowError, ValueError, TypeError):
            return match.group(0)

    return NUMBER.sub(spell, text)


def verbalize_status(sidecars, config):
    """What verbalization will manage on this batch: its languages, and what it cannot spell.

    The aligner itself is only borrowed in phase 2, so year disambiguation is announced from the
    config rather than from whether it has been picked up yet.
    """

    # strided rather than the first 200, which would all come from the same shard
    sample = sidecars[::max(1, len(sidecars) // 200)][:200]
    languages = {json.loads(path.read_text())["language"] for path in sample}
    unspellable = sorted(language for language in languages if not spellable(language))
    status = f"on ({', '.join(sorted(languages))})"
    if unspellable:
        status += f"{Colors.WARNING} — num2words has no {', '.join(unspellable)}{Colors.ENDC}"
    if not config.ctc_enabled:
        status += f"{Colors.WARNING} — no CTC stage, years read as counts{Colors.ENDC}"
    return status


def spellable(language):
    try:
        num2words(0, lang=language)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------
# Phase 2 — filtering: the metric cascade, then the multi-speaker source pass it feeds
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    rows: list                          # accepted clips, as CSV rows
    accepted: list                      # (speaker, duration) per accepted clip, for the report
    verdicts: collections.Counter
    failures: int
    rejected_sample: Reservoir
    accepted_sample: Reservoir


def filter_clips(sidecars, config, args):
    """Every clip through the cascade, then the sources the multi-speaker stage distrusts."""

    pipeline = Pipeline(config)
    prepare_verbalizer(config, pipeline)               # the dump helpers load() in this process
    order = [stage.verdict for stage in pipeline.stages]
    source_pass = config.source_rejection_enabled and QualityVerdict.MULTI_SPEAKER in order
    print_info("source pass", f"reject a speaker above {config.source_max_flag_rate:.0%} flagged "
                              f"clips (min {config.source_min_clips})" if source_pass else "off")

    out = Outcome([], [], collections.Counter(), 0,
                  Reservoir(DUMP_SAMPLE, args.seed), Reservoir(DUMP_SAMPLE, args.seed))
    scored_clips, flagged_clips = collections.Counter(), collections.Counter()
    start = time.time()
    for index, (path, verdict, error, duration, meta) in enumerate(
        scored(sidecars, config, args), 1
    ):
        if error:
            print(f"{Colors.FAIL}{path.stem}: {error}{Colors.ENDC}")
            out.failures += 1
            continue

        out.verdicts[verdict] += 1
        if reached(verdict, order):
            scored_clips[meta["speaker"]] += 1
            flagged_clips[meta["speaker"]] += verdict is QualityVerdict.MULTI_SPEAKER
        if verdict is QualityVerdict.ACCEPTED:
            out.rows.append((DATASET, path.stem, meta["text"], meta["speaker"], meta["language"]))
            out.accepted.append((meta["speaker"], duration))
            out.accepted_sample.add((path, duration, meta["speaker"]))
        elif verdict not in (QualityVerdict.TOO_SHORT, QualityVerdict.TOO_LONG):
            # length is a sourcing decision rather than a defect, so those are never dumped
            out.rejected_sample.add((path, verdict, duration))
        if index % 200 == 0:
            rate = index / (time.time() - start)
            print(f"  {index}/{len(sidecars)} clips, {rate:.1f}/s, "
                  f"{100 * len(out.rows) / index:.0f}% accepted", flush=True)

    if source_pass:
        sources = flagged_sources(scored_clips, flagged_clips, config)
        dropped = drop_sources(sources, out)
        out.verdicts[QualityVerdict.ACCEPTED] -= dropped
        out.verdicts[QualityVerdict.MULTI_SPEAKER_SOURCE] += dropped
        print_info("sources rejected", f"{len(sources)} of {len(scored_clips)} speakers, "
                                       f"{dropped} of their clips")
    return out


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


def drop_sources(sources, out):
    """Remove every clip of a rejected source, listening sample included; return how many."""

    dropped = len(out.rows)
    out.rows = [row for row in out.rows if row[3] not in sources]
    out.accepted = [item for item in out.accepted if item[0] not in sources]
    out.accepted_sample.items = [item for item in out.accepted_sample.items
                                 if item[2] not in sources]
    return dropped - len(out.rows)


def scored(sidecars, config, args):
    """Verdicts for every clip, in a worker pool unless --workers 1."""

    if args.workers <= 1:
        _setup(config, args.verbose)
        yield from (_score(path) for path in sidecars)
        return
    with multiprocessing.Pool(args.workers, _setup, (config, args.verbose)) as pool:
        yield from pool.imap_unordered(_score, sidecars, chunksize=8)


_pipeline = None
_verbalize = False
_aligner = None


def _setup(config, verbose):
    """One pipeline per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline
    torch.set_num_threads(1)
    _pipeline = Pipeline(config, verbose=verbose)
    prepare_verbalizer(config, _pipeline)


def _score(path):
    try:
        audio, meta = load(path)
    except Exception as error:
        return path, None, str(error), 0.0, None
    verdict = _pipeline.run(audio, meta["text"])
    return path, verdict, None, len(audio["audio"]) / audio["sample_rate"], meta


def prepare_verbalizer(config, pipeline):
    """Borrow the CTC stage's loaded MMS_FA for the year decision, rather than a second copy.

    Leaves the decision unavailable exactly when that stage is switched off.
    """

    global _verbalize, _aligner
    _verbalize = config.verbalize_numbers_enabled
    _aligner = next((stage.metric for stage in pipeline.stages
                     if stage.verdict is QualityVerdict.CTC_ALIGNMENT), None)


# ---------------------------------------------------------------------------------------------
# I/O and listening dumps
# ---------------------------------------------------------------------------------------------


def source_of(path):
    """The speaker ID a clip belongs to, from its filename: EN_B00000_S00040_W000003."""

    return path.stem.rsplit("_W", 1)[0]


def load(path):
    """The clip's audio and its sidecar, with phase-1 verbalization already applied."""

    audio, sample_rate = sf.read(path.with_suffix(".mp3"), dtype="float32")
    clip = {"audio": audio, "sample_rate": sample_rate}
    meta = json.loads(path.read_text())
    if _verbalize:
        meta["text"] = verbalize(meta["text"], meta["language"], clip)
    return clip, meta


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        # No Emilia transcript contains SEPARATOR or a backslash, so the writer never escapes;
        # quotechar=None leaves the ASR's own quotes as written.
        writer = csv.writer(handle, delimiter=SEPARATOR, quotechar=None,
                            quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def dump_rejected(sample, config):
    """Copy each sampled clip next to a note of what it scored and what was allowed.

    Scores are recomputed for the sample only — describing every rejected clip would re-evaluate
    a metric the pipeline had already decided on.
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


if __name__ == "__main__":
    main()
