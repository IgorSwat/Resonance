# Resonance refactor plan

Status: implemented on branch `refactor/pipeline` (2026-11-14), except the items in
[section 13](#13-not-done). The pre-refactor code is at git tag `pre-refactor`. Measured parity and speed are in
`knowledge/yodas2_sidon.md` §15-§16.

Goal: turn Resonance into a YODAS2-Sidon-first repo built around a modular, configurable pipeline that runs
unattended on one or more GPUs. The decisions it was built on are in [section 12](#12-decisions).

## 1. Current state

The facts the plan is built on, from commit `d12a1e7`.

- **Size:** 76 files, about 11,250 lines. The YODAS2-Sidon path is 10 scripts plus `tools/`, driven by the 98-line
  `scripts/pipeline/yodas2_sidon.sh`.
- **The YODAS path depends on other corpora's scripts.** Deleting them first would break it:
  - `scripts/filter/yodas2_sidon.py` imports `reached` from `filter/emilia.py`, `report` from `filter/libritts.py`
    and `BANDS` from `finalize/emilia_prune.py`;
  - `scripts/filter/yodas2_sidon_speaker_fixup.py` is mostly `filter/emilia_speaker_fixup.py`;
  - `scripts/fetch/yodas2_sidon.py` imports `extract` from `fetch/emilia.py` and `download`/`token` from
    `fetch/parla_speech_pl.py`;
  - select and finalize import from `select/libritts.py` and `finalize/libritts.py`.
- **ECAPA multi-speaker detection is already in the YODAS filter.** `SpeakerDriftMetric` runs as the filter's last
  pass and `speaker_drift_enabled` is true in the YODAS config. Nothing needs adding; it needs to become its own
  module.
- **Configuration is split three ways:** a flat YAML for filter thresholds (which also holds keys for Emilia and
  ParlaSpeech), argparse constants for segmenting and speaker fixup, and environment variables plus hardcoded
  `/workspace` paths in the bash driver. A wrong `--config` path silently falls back to built-in defaults.
- **Failures are silent.** A metric that throws rejects the clip. A missing HF token for the gated
  `pyannote/segmentation-3.0` rejected every clip of a run on 2026-11-12 without any crash.
- **No multi-GPU support.** Two drivers are started by hand, one per GPU. Their downloaders share one raw directory
  and count its folders to bound look-ahead, so leftover raw batches block both.
- **Resume state is keyed by run range** (`yodas_en001_91-105/`), so a batch processed in one run isn't recognised
  by another, and merging across runs is manual.
- **GPU use:** the filter scores one clip at a time in 6 processes, each holding every model (22-24 GB on a 24 GB
  card). Speaker fixup embeds clip by clip and compares speakers pair by pair in a Python loop.
- **Time per batch** (en001 batches 91-120, 2026-11-13, one GPU per half): about 7.5 min segmenting and about 3 min
  filtering. Segmenting is roughly 70% of the cost.
- **Disk per batch:** a raw batch is about 3 GB and is deleted after segmenting. The segmented clips stay, at about
  3.5 GB per English batch, and about half of them are rejected by the filter.
- **No tests.** `requirements.txt` is a 123-line freeze that misses `faster_whisper`.

## 2. Target layout

```
pyproject.toml                  direct dependencies; installed with `pip install .`
resonance/                      importable package
  config/        schema.py, loader.py
  log/           style.py (colors, banners, tables), console.py (tqdm bars, run.log), events.py, capture.py
  io/            manifest.py (the CSV dialect), audio.py (read, resample kernels)
  pipeline/      stage.py (contract), artifacts.py, registry.py, runner.py, state.py (markers, resume),
                 scheduler.py and gpu.py (GPU workers), pool.py, preflight.py, disk.py, cli.py
  stages/        fetch.py, segment.py, quality.py, source_pass.py, speaker_drift.py, discard_rejected.py,
                 merge.py, speaker_fixup.py, select.py, finalize.py
  segmentation/  vad.py, transcribe.py, sentences.py, packing.py, abbreviations.py
  metrics/       metric.py, one file per metric (as today)
  speakers/      embedding.py, clustering.py
  codecs/        codec.py (contract), registry.py, higgs.py
  datasets/      yodas2_sidon.py (HF repo layout, metadata.json reading)
configs/         yodas2_sidon.yaml
scripts/         setup.sh, run_pipeline.py, status.py, preflight.py, stages/<stage>.py
tests/
knowledge/
```

Rules:

- One class per file for stages, metrics and codecs.
- Pure helper functions are grouped by topic, one topic per file.
- Scripts hold argument parsing and a call into the package, nothing else.
- Every function has typed arguments and a typed return value.
- The 488-line segmenter splits into the `segmentation/` files plus a thin stage.

## 3. Modules and contracts

Two artifact types cover almost everything:

- **RawBatch:** a directory of `<n>.flac` plus `<n>.metadata.json`, one pair per video.
- **ClipManifest:** a CSV in the shared dialect plus the directory that holds its wavs. Between stages it carries
  three extra columns, `duration`, `audio_path` and `source_id`; `finalize` trims it to the five standard columns
  (`dataset|name|transcription|speaker_id|language`).

| Module | Scope | Input | Output | GPU |
|---|---|---|---|---|
| `fetch` | batch | batch reference (language, shard, index) | RawBatch | no |
| `segment` | batch | RawBatch | ClipManifest (`segment.csv` + wavs) | yes |
| `quality` | batch | ClipManifest | ClipManifest + `verdicts.csv` (every input clip: verdict and the scores behind it) | yes |
| `source_pass` | batch | ClipManifest + `verdicts.csv` | ClipManifest | no |
| `speaker_drift` | batch | ClipManifest | ClipManifest + drift scores | yes |
| `discard_rejected` | batch | ClipManifest + `segment.csv` | the same ClipManifest; the audio of rejected clips is deleted | no |
| `merge` | pool | list of batch manifests | one ClipManifest | no |
| `speaker_fixup` | pool | ClipManifest | ClipManifest with `speaker_id` rewritten + `merges.csv` | yes |
| `select` | pool | ClipManifest | ClipManifest | no |
| `finalize` | pool | ClipManifest | dataset directory: five-column CSV, `audio.zip`, optional `codecs/<codec>/` | only with codecs |

What each module does:

- **`fetch`** downloads and unpacks one batch from `sarulab-speech/yodas2_sidon`. It also re-checks free disk
  before each download (section 6).
- **`segment`** cuts each recording into sentence-aligned utterances and transcribes them with Whisper, as
  `scripts/segment/yodas2_sidon.py` does today. Every parameter moves into the config. The raw batch is deleted
  afterwards unless `keep_raw` is set.
- **`quality`** is one module that runs the per-clip cascade: duration bounds, digital silence, bandwidth, clipping,
  mains hum, spectral flatness, multi-speaker (pyannote), NISQA, CTC alignment. The metrics are an ordered,
  switchable list in the config. It stops at a clip's first failing metric.
- **`source_pass`** drops a whole recording when the multi-speaker metric rejected more than `max_flag_rate` of the
  clips it scored.
- **`speaker_drift`** scores each surviving clip with ECAPA and drops those at or below `max_z` within their
  duration band.
- **`discard_rejected`** deletes the wav of every clip the batch's filters rejected, about half of the batch.
  After it, the filters cannot rerun on the batch without segmenting it again.
- **`merge`** concatenates batch manifests into one pool.
- **`speaker_fixup`** merges video IDs that are one person and rewrites `speaker_id`.
- **`select`** applies simple rules: minimum clips per speaker, maximum clips per speaker, maximum minutes per
  speaker, seeded random choice within the caps.
- **`finalize`** packs the selected clips into the final product (below).

### The stage contract

- Each stage is a class with `name`, `scope` (`batch` or `pool`), `consumes`, `produces`, `needs_gpu` and
  `run(input, ctx) -> output`.
- The pipeline is the ordered list of stage names in the config.
- At startup the runner checks that each stage's input type matches the previous stage's output, so a bad
  arrangement fails before any work starts.
- Adding a module takes one file in `stages/`, one registry entry and one config section.

### The metric contract

- A metric returns scores, and the cascade compares them to the bounds. `verdicts.csv` records the scores, so
  thresholds can be re-tuned without rerunning the GPU work.
- An exception inside a metric is recorded as the verdict `error`, never as a rejection.
- Auth (401), CUDA and out-of-memory errors abort the stage at once. Other errors abort it when their share passes
  `quality.max_error_rate`.

### Finalize

Active by default. Its product is `runs/<run name>/dataset/`:

```
<run name>.csv        five standard columns
audio.zip             <name>.wav for every selected clip
codecs/<codec>/       only when codecs are enabled: <name>.npy per clip
```

- **Clips are moved, not copied.** The selected wavs are renamed out of `clips/` into `dataset/audio/`, which costs
  no extra disk. The zip therefore holds a part of what the earlier stages produced, and those clips no longer
  exist in `clips/`.
- **Then zipped.** `audio/` is written to `audio.zip` with deflate compression, which is lossless, the zip is
  verified (file count and CRC check), and only then is `audio/` removed. While the zip is being written the selected set exists twice on disk.
- **No codec tokens by default.** `finalize.codecs` is an empty list. Naming a codec (`[higgs]`) encodes every
  clip before the zip step and writes `codecs/higgs/<name>.npy`. A new codec is one class in `resonance/codecs/`
  behind the `Codec` contract (`name`, `sample_rate`, `encode(audio) -> tokens`) plus a registry entry, and gets its
  own subdirectory.
- **Finalize is the one irreversible stage.** After it, the batches it drew from are marked as consumed by that
  run. A later run that includes a consumed batch stops with a message naming the run, because the manifests of
  that batch point at clips that moved. `scripts/stages/finalize.py --restore` unzips the clips back to where they
  came from.
- Clips that were not selected stay in `clips/`, so a second run with other selection rules can still use them.
- `finalize.move: false` copies instead, `finalize.zip: false` leaves `audio/` as a directory, and
  `finalize.compression: stored` skips compression.

## 4. Configuration

One YAML file per pipeline, loaded into typed dataclasses with PyYAML. No new dependency.

- An unknown key is an error. A missing config file is an error.
- The resolved config is saved into the run folder.
- Any key can be overridden on the command line: `--set quality.nisqa.min.noisiness=3.2`.
- The calibration comments of `configurations/quality_filtering_yodas2_sidon.yaml` carry over.

```yaml
run:
  name: en_81_132
  data_root: data/yodas2_sidon

source:
  language: en
  batches:
    - {shard: "001", first: 81, last: 132}
    - {shard: "002", first: 0, last: 19}

resources:
  gpus: all                 # or [0, 1]
  fetch_ahead: 3            # raw batches waiting on disk, across all GPUs
  cpu_workers: 16
  threads_per_worker: 4

disk:
  raw_gb_per_batch: 3.0     # used only for the estimate
  clips_gb_per_batch: 3.5
  kept_share: 0.5           # share of the segmented audio that survives the filters

pipeline: [fetch, segment, quality, source_pass, speaker_drift, discard_rejected,
           merge, speaker_fixup, select, finalize]

segment:
  whisper: large-v3-turbo
  batch_size: 16
  beam_size: 5
  target_duration: 8.0
  min_duration: 3.0
  max_duration: 25.0
  max_pause: 2.0
  min_gap: 0.15
  lead: 0.1
  trail: 0.2
  keep_raw: false

quality:
  min_duration: 3.0
  max_duration: 30.0
  metrics: [bandwidth, clipping, hum, flatness, multi_speaker, nisqa, ctc]   # order = cascade order
  bandwidth: {min_frac: 0.85, band_hz: 1000.0, cliff_db: 20.0}
  clipping: {max_ratio: 0.0001}
  hum: {max_prominence: 15.0, freq_tolerance: 0.3}
  flatness: {min: 0.0005, max: 0.25, loudness_threshold: 35.0}
  multi_speaker: {max: {second: 0.0, overlap: 0.0}}
  nisqa: {min: {noisiness: 3.0, discontinuity: 3.5, coloration: 3.25, loudness: 3.0}}
  ctc: {uroman: true, max: {loss: 0.7, max_loss: 1.0, maxrate: 30.0, gap_speech: 0.06, trail_speech: 0.05}}
  max_error_rate: 0.01

source_pass: {enabled: true, max_flag_rate: 0.3, min_clips: 3}
speaker_drift: {enabled: true, max_z: -2.0}
speaker_fixup: {threshold: 0.5, linkage: single, clips_per_speaker: 4}
select: {min_per_speaker: 4, max_per_speaker: 64, max_minutes_per_speaker: null, seed: 0}
finalize: {move: true, zip: true, compression: deflate, codecs: []}    # codecs: [higgs] -> codecs/higgs/<name>.npy
```

## 5. Runner, resume and multiple GPUs

Layout on disk, all under `run.data_root`:

```
raw/<lang>/<shard>/train-00091/            deleted after segmenting unless keep_raw
clips/<lang><shard>/train-00091/*.wav      selected clips leave here at finalize
batches/<lang><shard>/train-00091/
    segment.csv  quality.csv  verdicts.csv  source_pass.csv  speaker_drift.csv  speaker_drift_scores.csv
    discard_rejected.csv  <stage>.done  logs/<stage>.log
runs/<run name>/
    config.yaml  run.log  events.jsonl
    merge.csv  speaker_fixup.csv  merges.csv  select.csv
    dataset/   <run name>.csv  audio.zip  [codecs/<codec>/]
```

- **State is keyed by batch.** A stage's `.done` marker is written atomically after its output. A later run over a
  wider range reuses the batches that are already done.
- **Pool stages** (merge onward) live in `runs/<run name>/`.
- **Scheduling:** one worker process per GPU, spawned with its own `CUDA_VISIBLE_DEVICES` and thread caps. Workers
  pull batches from a shared queue, so an uneven batch doesn't leave a GPU idle.
- **Fetching:** one downloader feeds the queue and bounds look-ahead by its own count, not by listing a directory.
- **Pool stages on several GPUs:** speaker-fixup embedding and codec encoding split their clip list across GPUs.
- **Preflight before any work:** HF token and gated-model access, `aria2c`, visible GPUs, imports, and the disk
  budget of section 6.
- **Resume:** rerun the same command. `--from <stage>` forces that stage and everything after it to rerun.

## 6. Disk budget warning

The pipeline estimates the disk a run needs and warns when the requested batches don't fit.

- **Estimate:** batches still to process × `clips_gb_per_batch` × `kept_share` (rejected audio is discarded), plus
  raw batches and unfiltered clips in flight, plus the kept clips of the whole run once more for the zip step.
- **At start (preflight):** the estimate is compared with free space on `data_root`. When it doesn't fit, a warning
  banner shows the estimate, the free space and how many of the requested batches fit. The run continues.
- **During fetch:** before each download the same check runs against the actual free space and the actual average
  clip size of the batches finished so far, so the warning gets more accurate as the run goes.
- **Example:** 100 batches need about 370 GB with the zip step. On a 256 GB disk the warning fires at once; about
  65 batches fit.
- It is only ever a warning: the run does not stop at a floor of free space.

## 7. Scripts

- **`bash scripts/setup.sh`** prepares a box with one command:
  1. asks for the HF token (hidden input) and stores it in the Hugging Face cache with owner-only permissions; it
     skips the question when a token is already stored or `HF_TOKEN` is set;
  2. installs `aria2` with `apt install -y aria2`;
  3. installs Resonance and its dependencies with `pip install .`;
  4. runs the preflight checks, prints what is still missing, and downloads every model the pipeline uses, so
     the first run's workers don't all fetch the same file at once.

  The install is not editable, so `setup.sh` (or `pip install .`) is run again after pulling new code.
- **`scripts/run_pipeline.py --config configs/yodas2_sidon.yaml`** runs the whole pipeline and needs no attention.
  Flags: `--set key=value`, `--from <stage>`, `--until <stage>`, `--dry-run` (prints the plan, the disk estimate and
  what is already done).
- **`scripts/stages/<stage>.py`** runs one stage on one batch or one pool, with the same config file. The
  `select` stage's script is `select_clips.py`, because a `select.py` would shadow Python's own module.
- **`scripts/status.py <run>`** prints a batches × stages table with clips and hours per stage.
- **`scripts/preflight.py`** runs the environment checks alone.

Every script has a docstring with a usage example and a complete `--help`.

## 8. Logging

- `resonance/log/style.py` replaces `scripts/__style__.py`: stage banners, key/value lines, summary tables, warning
  and error styles. The five unused test helpers are dropped.
- Workers never print. They send events (stage start, progress, error, stage summary) to the parent process.
- The parent draws one tqdm bar per GPU plus an overall batch bar.
- Per-clip lines disappear. Only exceptions and errors are logged individually.
- Each stage ends with a summary: clips in, clips out, hours, rejections by reason, errors, wall time.
- Everything also goes to `run.log` as plain text and to `events.jsonl`, so `status.py` and external watchers never
  parse progress bars.

## 9. Optimization

In order of expected value. Each item is measured before and after on the same batch. No speedup is promised for
items 2-5 until they are measured.

1. **Multi-GPU scheduling** (section 5). Replaces the manual split into halves.
2. **Segmenting**, which is about 70% of batch time. The segmenter already times read, resample, VAD, waiting and
   Whisper separately. The first step is to read those numbers and fix whichever dominates. Candidates: more prep
   workers, a larger Whisper batch, batching windows across recordings.
3. **Batched quality metrics:** one process per GPU with each model loaded once, clips decoded by CPU workers and
   scored in length-bucketed batches, the cascade still dropping rejects between metrics. Accepted when at least
   99.5% of verdicts on the parity batch are identical to the per-clip path.
4. **Speaker fixup:** batched embedding, and one matrix product in place of the per-pair Python loop.
5. **Codec encoding** (only when a codec is enabled): batched, only if the tokens stay identical.

## 10. What gets removed

- All fetch, filter, select and finalize scripts for Emilia, LibriTTS, ParlaSpeech, WolneLektury and CML-TTS, and
  their six configs. The helpers the YODAS path borrows from them move into the package first.
- The ΔCE scorer and selector (`scripts/score/yodas2_sidon_dce.py`, `scripts/select/yodas2_sidon_dce.py`), the
  prosody selection (`scripts/select/yodas2_sidon.py`) and `tools/prosody/`.
- `tools/load/` (nothing imports it) and the empty `tools/metrics/constants.py`.
- The non-YODAS keys of `QualityConfig`: digit rejection, output loudness, number verbalisation, duplicate sources.
- `scripts/pipeline/yodas2_sidon.sh`, replaced by the runner.
- `scripts/finalize/yodas2_sidon_prune.py`: the drift pass already runs inside the filter.
- `requirements.txt`, replaced by `pyproject.toml` with direct dependencies (including `faster-whisper`).
- Knowledge notes of the removed corpora: `knowledge/emilia.md`, `knowledge/wolne_lektury.md` and
  `knowledge/cml_tts.md`. Two findings in `emilia.md` are what the YODAS drift pass and speaker fixup rest on
  (similarity falls with clip length, so drift is judged by z within duration bands; sources are compared by
  clip-pair agreement, not by centroids). They move into `knowledge/yodas2_sidon.md` before the note is deleted.
  `knowledge/data_filtering.md` stays: it is the calibration of the cascade YODAS uses.

A `pre-refactor` git tag keeps all of it recoverable.

## 11. Verification and phases

### Verification

- **Unit tests** for logic that needs no GPU: sentence splitting and packing, the source pass, drift z-bands,
  speaker clustering, selection rules, manifest round-trips (including `|` and a bare `\r` in the text), config
  strictness, the disk estimate, finalize's move, zip and restore.
- **Runner tests** with fake stages: ordering, the type check, resume, consumed batches, scheduling across fake
  GPUs.
- **Parity run on a GPU box:** one batch through the old code and the new, comparing clip lists, transcripts,
  verdict counts and speaker merges. The structural phases must match exactly; batched metrics must reach 99.5%
  identical verdicts.

The parity run needs a GPU box with one raw batch and the golden outputs of the current code for it.

### Phases

Each phase leaves a working pipeline and is one reviewable PR.

1. **Baseline:** tag `pre-refactor`; record the golden outputs of one batch with the current code.
2. **Prune:** move the borrowed helpers, delete the other corpora, the ΔCE and prosody code and the unrelated
   knowledge notes. Behaviour of the YODAS path unchanged.
3. **Foundation:** package, `pyproject.toml`, `setup.sh`, config loader, log style, manifest IO, test setup.
4. **Stages and runner:** port each stage behind the contract on a single GPU with exact parity. Split the filter
   into `quality`, `source_pass` and `speaker_drift`. Change the metric contract (scores and `error` verdicts).
   Write the new `select` and `finalize`.
5. **Scripts and observability:** top-level and per-stage scripts, status, preflight, disk warning, event log.
6. **Multi-GPU scheduler.**
7. **Optimization** (section 9), one item at a time, each with measurements.
8. **Docs:** rewrite `CLAUDE.md`, the `run-pipeline` skill and `knowledge/yodas2_sidon.md`; log the change in
   Echo's `knowledge/data_processing/`.

## 12. Decisions

All by the user, 2026-11-14.

- **Filter granularity:** one `quality` module with an ordered, switchable list of metrics, plus separate
  `source_pass` and `speaker_drift` modules.
- **Manifest between stages:** the CSV dialect with extra columns, trimmed to the five standard ones at the end.
- **Metric failures:** an exception is recorded as `error`, not as a rejection.
- **Selection:** only the rule-based `select`. The ΔCE scorer and selector and the prosody selection are deleted.
- **Finalize:** active by default. It moves the selected clips, zips them to `audio.zip` and removes `audio/`. No
  codec tokens by default; `codecs/<codec>/` stays as the extensible place for them.
- **Parity of batched metrics:** at least 99.5% identical verdicts on the parity batch.
- **Config library:** plain dataclasses plus PyYAML.
- **Knowledge of removed corpora:** deleted.
- **Package name:** `resonance/`.
- **Setup:** `scripts/setup.sh` asks for the HF token, installs `aria2` with apt and runs `pip install .`.
- **Disk:** a warning when the requested batches are too many for the disk, and only a warning.
- **Rejected clips:** their audio is deleted (the `discard_rejected` stage).
- **Unselected clips:** kept after finalize.
- **Zip compression:** deflate, since it is lossless.
- **Expressiveness, emotion and voice-uniqueness scorers:** not part of this refactor; maybe later.

## 13. Not done

Added after the plan: `segment.workers` runs several Whisper loops on one GPU (default 3), which cut segmenting by
30% with identical output, and `resources.cpu_workers` defaults to 24 (`knowledge/yodas2_sidon.md` §16).

- **Batched quality metrics** (section 9, item 3). The quality stage still scores one clip at a time in 6 worker
  processes per GPU, as the old filter did, with identical verdicts. Segmenting is about 70% of a batch's time and
  Whisper most of that, so this was not where the time is; it stays open as an optimization with the 99.5% parity
  bar.
- **Batched codec encoding** (section 9, item 5). Codec encoding is split across GPUs but encodes clip by clip.
- **One PR per phase** (section 11). The work is three commits on one branch: the new package, the removal of the
  old code, and the docs.
