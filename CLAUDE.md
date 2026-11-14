# Resonance

Resonance turns YODAS2-Sidon (Sidon-restored YouTube speech) into TTS training data. One configurable pipeline
downloads batches, cuts them into sentence-aligned clips with Whisper transcripts, filters them through a quality
cascade, fixes speaker labels, selects clips and packs the result. The product of a run is
`<data_root>/runs/<run>/dataset/`: a `|`-separated CSV (`dataset|name|transcription|speaker_id|language`),
`audio.zip`, and optionally `codecs/<codec>/<name>.npy`.

The cascade's design and calibration are in `knowledge/data_filtering.md`; what the corpus contains and how the
pipeline was measured is in `knowledge/yodas2_sidon.md`.

## Layout

- `resonance/` is the package. `pipeline/` holds the stage contract, the runner, the multi-GPU scheduler, resume
  state, preflight and the disk estimate. `stages/` has one file per pipeline module. `metrics/`, `segmentation/`,
  `speakers/` and `codecs/` hold what the stages use. `config/` is the schema and the strict loader, `log/` the
  console and log style, `io/` the manifest format and audio reading.
- `configs/yodas2_sidon.yaml` is the pipeline config: every stage's parameters, the batches to process, and the
  stage order. An unknown key or a missing file is an error.
- `scripts/run_pipeline.py` runs everything; `scripts/stages/<stage>.py` runs one stage (`select_clips.py` for `select`); `scripts/status.py` shows
  progress; `scripts/preflight.py` checks the machine; `scripts/setup.sh` prepares a new one.
- `tests/` runs with `python -m pytest` and needs no GPU.
- `data/` and `tmp/` are gitignored. Everything a run writes lives under `run.data_root`.

The pipeline, in order:

| stage | scope | does |
|---|---|---|
| `fetch` | batch | downloads and unpacks one batch (about 3 GB) |
| `segment` | batch | VAD, Whisper, sentence packing: writes clips and `segment.csv` |
| `quality` | batch | the metric cascade: writes `quality.csv` and `verdicts.csv` (every clip's verdict and scores) |
| `source_pass` | batch | drops recordings the multi-speaker metric flagged too often |
| `speaker_drift` | batch | drops clips whose voice changes (ECAPA, z within duration bands) |
| `discard_rejected` | batch | deletes the audio of rejected clips |
| `merge` | pool | concatenates the batches of the run |
| `speaker_fixup` | pool | merges video IDs that are one person |
| `select` | pool | per-speaker minimum and caps |
| `finalize` | pool | moves the selected clips into `dataset/`, zips them, optionally encodes codec tokens |

Batch stages keep their state per batch (`<data_root>/batches/<shard>/<batch>/<stage>.done`), so a later run over
more batches reuses the ones already done. Pool stages keep theirs in `<data_root>/runs/<run>/`.

## Running

```bash
bash scripts/setup.sh                                               # once per machine
python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml --dry-run
python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml   # inside tmux; rerun the same command to resume
python scripts/status.py --config configs/yodas2_sidon.yaml
```

Run from the repo root with Python 3.12 or later. `pip install .` is not editable: reinstall after changing code,
or run with `PYTHONPATH=.` while developing. The runner uses every GPU `nvidia-smi` lists, one worker per GPU.

## Remote GPU boxes

Segmentation and filtering need a CUDA box. The box changes often (usually a vast.ai instance), so ask the user
which one to use for each task. Each user keeps their own SSH aliases and checkout paths in `CLAUDE.local.md`
(gitignored).

## ⚑ Gotchas

- **`finalize` moves clips.** After it, the selected wavs exist only in `audio.zip`, and the batches they came from
  are marked consumed by that run. `scripts/stages/finalize.py --restore` puts them back.
- **`discard_rejected` deletes audio.** After it, the filters cannot rerun on a batch without `--from fetch`.
- **A metric's exception is an error, not a rejection.** Auth, CUDA and out-of-memory errors stop the stage at
  once; other errors stop it above `quality.max_error_rate`. A metric raises `Unscorable` when the clip itself is
  the problem, and only that counts as a rejection.
- **Use spawn, not fork, for any process that touches CUDA** (`resonance/pipeline/gpu.py`, `pool.py`).
- **Quality workers set TF32 matmuls on purpose.** Every threshold was calibrated that way; see
  `knowledge/yodas2_sidon.md` §15.
- **Read and write manifests only through `resonance/io/manifest.py`.** Rows end in CRLF and transcripts can
  contain a bare `\r`, so `awk`, `cut`, `head` and `paste` split rows.
- **A dict-valued config key replaces its default wholesale** (`quality.nisqa.min`, `quality.ctc.max`).
- **NISQA on MPS returns wrong scores**, and only CUDA reproduces existing codec tokens. A Mac is for tests only.
