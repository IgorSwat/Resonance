"""Filter the YODAS2-Sidon utterances cut by scripts/segment/yodas2_sidon.py.

    python scripts/filter/yodas2_sidon.py --input segments.csv --audio-dir <segment wavs>

The segmenter has already cut every recording into sentence-aligned utterances and transcribed
them with Whisper, so this script only judges them, in four steps:

1. **The cascade** of tools/metrics/pipeline.py — DSP stages, pyannote multi-speaker, NISQA, and
   last CTC against the segmenter transcript.
2. **ECAPA speaker drift** is scored on every clip the cascade accepts.
3. **The source pass** drops a whole recording when the multi-speaker stage rejected more than
   config.source_max_flag_rate of the clips it scored: the per-clip stage cannot notice an
   interview whose speakers fall cleanly into separate clips. Note the asymmetry with Emilia: a
   speaker there is one voice in one recording, while a video here is one recording that may hold
   several, so dropping a source discards its good voices along with its bad.
4. **The speaker-drift pass** is the last guard against a second voice. SpeakerDriftMetric asks
   whether the voice itself changes between two moments of the clip — the case the diarisation
   stage misses when two similar voices take clean turns. Its similarity falls with clip length,
   so a clip is flagged by z within its duration band, as scripts/finalize/*_prune.py do, which
   needs every survivor scored first.

Steps 1 and 2 run whole in each worker of a spawned pool: a worker reads its clip, runs every model
on the GPU and sends back only the verdict, the duration and the drift score. The parent holds no
model and no audio, so it never becomes the bottleneck. Results come back in order rather than as
they finish, because the source pass needs a recording's clips together.

No audio is written: the accepted rows point at the segmenter's clips, which is where
scripts/filter/yodas2_sidon_speaker_fixup.py and the later stages read them.
"""

import argparse
import collections
import csv
import dataclasses
import multiprocessing
import pathlib
import random
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.__style__ import Colors, print_info, print_section, print_test_title
from scripts.filter.emilia import reached
from scripts.filter.libritts import report
from scripts.finalize.emilia_prune import BANDS
from scripts.segment.yodas2_sidon import AUDIO, COLUMNS, DATASET, SEPARATOR
from tools.metrics.pipeline import Pipeline
from tools.metrics.speaker_drift import SpeakerDriftMetric
from tools.metrics.types import QualityConfig, QualityVerdict

CONFIG = pathlib.Path("configurations/quality_filtering_yodas2_sidon.yaml")
DIALECT = {"delimiter": SEPARATOR, "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
PROGRESS_EVERY = 100
WORKERS = 6                     # the GPU saturates by 6; at 10 the workers run it out of memory, and a metric that
                                # fails on OOM rejects the clip silently


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=pathlib.Path, required=True,
                        help="a CSV from scripts/segment/yodas2_sidon.py")
    parser.add_argument("--audio-dir", type=pathlib.Path, default=AUDIO,
                        help="where the segmenter wrote its clips")
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("yodas2_sidon_filtered.csv"))
    parser.add_argument("--limit", type=int,
                        help="filter the clips of N random recordings instead of every one")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed used by --limit")
    parser.add_argument("--verbose", action="store_true", help="print why each clip was rejected")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="processes, each holding every model on the GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QualityConfig.from_yaml(args.config) if args.config.exists() else QualityConfig()
    rows = read_rows(args.input, args.limit, args.seed)

    print_test_title(f"Filtering {DATASET}: {len(rows)} utterances, "
                     f"{len({row['speaker_id'] for row in rows})} recordings")
    print_info("input", args.input)
    print_info("audio", args.audio_dir)
    print_info("config", args.config if args.config.exists() else "built-in defaults")
    print_info("output", args.output)
    print_info("workers", args.workers)
    # Built only for its stage order; the metrics load their models lazily, so this costs nothing.
    order = [stage.verdict for stage in Pipeline(config).stages]
    print_info("cascade", ", ".join(verdict.value for verdict in order))
    print_info("ctc gate", f"{config.ctc_max}" if config.ctc_enabled else "off")
    print_info("speaker drift", f"flag z <= {config.speaker_drift_max_z} within duration bands"
                                if config.speaker_drift_enabled else "off")
    print_info("source pass", f"drop a recording above {config.source_max_flag_rate:.0%} flagged "
                              f"clips (min {config.source_min_clips} scored)"
                              if config.source_rejection_enabled else "off")

    print_section("Filtering")
    verdicts = collections.Counter()
    survivors, dropped, failures, start = [], [], 0, time.time()
    source = None

    for row, verdict, error, duration, similarity in scored(rows, config, args):
        if source is not None and source.video != row["speaker_id"]:
            resolve(source, survivors, dropped, verdicts, config)
            source = None
        if source is None:
            source = Source(row["speaker_id"])
        if error:
            print(f"{Colors.FAIL}{row['name']}: {error}{Colors.ENDC}", flush=True)
            failures += 1
            continue

        verdicts[verdict] += 1
        # whether the diarisation stage scored this clip, which is what the source tally needs
        if reached(verdict, order):
            source.scored += 1
            source.flagged += verdict is QualityVerdict.MULTI_SPEAKER
        if verdict is QualityVerdict.ACCEPTED:
            source.clips.append(Clip(row, duration, similarity))
        done = sum(verdicts.values())
        if done % PROGRESS_EVERY == 0:
            kept = len(survivors) + len(source.clips)
            print(f"  {done}/{len(rows)} utterances ({100 * done / len(rows):.0f}%), "
                  f"{done / (time.time() - start):.1f}/s, {100 * kept / done:.0f}% accepted",
                  flush=True)

    if source is not None:
        resolve(source, survivors, dropped, verdicts, config)
    if dropped:
        print_info("sources rejected",
                   f"{len(dropped)} recordings over {config.source_max_flag_rate:.0%} flagged, "
                   f"{sum(len(item.clips) for item in dropped)} of their clips")

    if config.speaker_drift_enabled:
        flagged = drifting(survivors, config.speaker_drift_max_z)
        survivors = [clip for clip in survivors if clip.row["name"] not in flagged]
        verdicts[QualityVerdict.ACCEPTED] -= len(flagged)
        verdicts[QualityVerdict.SPEAKER_DRIFT] += len(flagged)
        print_info("speaker drift", f"{len(flagged)} clips at z <= {config.speaker_drift_max_z}")

    write(args.output, [clip.row for clip in survivors])
    accepted = [(clip.row["speaker_id"], clip.duration) for clip in survivors]
    report(verdicts, accepted, failures, time.time() - start, args.output)


def read_rows(path: pathlib.Path, limit: int | None, seed: int) -> list:
    """The segmenter's rows, grouped by recording, optionally only those of --limit recordings."""

    if not path.exists():
        raise SystemExit(f"No input at {path}; run scripts/segment/yodas2_sidon.py first")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, **DIALECT))
    if limit:
        videos = sorted({row["speaker_id"] for row in rows})
        keep = set(random.Random(seed).sample(videos, min(limit, len(videos))))
        rows = [row for row in rows if row["speaker_id"] in keep]
    # stable, so a recording's clips stay in the segmenter's order
    return sorted(rows, key=lambda row: row["speaker_id"])


def scored(rows: list, config: QualityConfig, args: argparse.Namespace):
    """(row, verdict, error, duration, similarity) for every clip, in a worker pool unless --workers 1."""

    tasks = [(row, args.audio_dir) for row in rows]
    if args.workers <= 1:
        _setup(config, args.verbose)
        for task, outcome in zip(tasks, map(_score, tasks)):
            yield task[0], *outcome
        return

    # spawn, never the fork Linux defaults to: a child forked from a process that has touched
    # CUDA inherits a context it cannot use, and MultiSpeakerMetric answers that failure with a
    # rejection — so the run would quietly reject every clip rather than fail.
    with multiprocessing.get_context("spawn").Pool(
        args.workers, _setup, (config, args.verbose)
    ) as pool:
        for task, outcome in zip(tasks, pool.imap(_score, tasks, chunksize=4)):
            yield task[0], *outcome


_pipeline = None
_drift = None


def _setup(config: QualityConfig, verbose: bool) -> None:
    """Every model once per worker. Single-threaded torch, or the processes fight over cores."""

    global _pipeline, _drift
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    _pipeline = Pipeline(config, verbose=verbose)
    if config.speaker_drift_enabled:
        _drift = SpeakerDriftMetric(device=config.speaker_drift_device)


def _score(task: tuple):
    """One clip through the cascade, and its drift score when accepted: (verdict, error, duration,
    similarity). A digitally silent clip is answered without the cascade."""

    row, audio_dir = task
    try:
        y, rate = sf.read(audio_dir / f"{row['name']}.wav", dtype="float32")
        audio = {"audio": y.mean(axis=1) if y.ndim > 1 else y, "sample_rate": rate}
        duration = len(audio["audio"]) / rate
        if audio["audio"].size and np.ptp(audio["audio"]) == 0:
            return QualityVerdict.SILENCE, None, duration, None
        verdict = _pipeline.run(audio, row["transcription"])
        similarity = None
        if verdict is QualityVerdict.ACCEPTED and _drift is not None:
            similarity = _drift.evaluate(audio)["similarity"]
        return verdict, None, duration, similarity
    except Exception as error:
        return None, str(error), 0.0, None


@dataclasses.dataclass
class Clip:
    row: dict
    duration: float
    similarity: float | None        # SpeakerDriftMetric's score, None when the pass is off


@dataclasses.dataclass
class Source:
    """One recording's tally, held until its last clip has been judged."""

    video: str
    clips: list = dataclasses.field(default_factory=list)
    scored: int = 0                 # clips the diarisation stage actually scored
    flagged: int = 0                # of those, the ones it rejected


def resolve(source: Source, survivors: list, dropped: list, verdicts: collections.Counter,
            config: QualityConfig) -> None:
    """Keep a finished recording's clips, or drop the source whole.

    The rate is taken over the clips that reached the diarisation stage, not over every clip: one
    rejected on length was never diarised and says nothing about the recording.
    """

    if not flagged_source(source, config):
        survivors.extend(source.clips)
        return
    verdicts[QualityVerdict.ACCEPTED] -= len(source.clips)
    verdicts[QualityVerdict.MULTI_SPEAKER_SOURCE] += len(source.clips)
    dropped.append(source)


def flagged_source(source: Source, config: QualityConfig) -> bool:
    """Whether the diarisation stage rejected enough of this recording to distrust all of it."""

    return (config.source_rejection_enabled
            and source.scored >= config.source_min_clips
            and source.flagged / source.scored >= config.source_max_flag_rate)


def drifting(clips: list, max_z: float) -> set:
    """Names whose drift similarity is at or below max_z within their duration band.

    The scan of scripts/finalize/emilia_prune.py over the filter's own survivors: z rather than a
    flat bound, since raw similarity falls with clip length (knowledge/emilia.md §3).
    """

    durations = np.array([clip.duration for clip in clips])
    scores = np.array([clip.similarity for clip in clips])
    z = np.zeros_like(scores)
    for low, high in BANDS:
        band = (durations >= low) & (durations < high)
        # A band with one clip has no spread to normalize by, and is left at z = 0.
        if band.sum() > 1:
            z[band] = (scores[band] - scores[band].mean()) / scores[band].std()
    return {clip.row["name"] for clip, value in zip(clips, z) if value <= max_z}


def write(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, **DIALECT)
        writer.writerow(COLUMNS)
        writer.writerows([row[column] for column in COLUMNS] for row in rows)


if __name__ == "__main__":
    main()
