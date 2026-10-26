# Resonance

Resonance turns public speech corpora into TTS training data: it downloads a corpus, filters it through a quality
cascade, selects a prosodically varied subset, and writes the audio with Higgs Audio v2 codec tokens. The output is a
`|`-separated CSV (`dataset|name|transcription|speaker_id|language`), `audio/`, and `codecs/<name>.npy` (int16
`[8, T]`) under `data/processed/<Corpus>/`.

The cascade's design and calibration are in `knowledge/data_filtering.md`.

## Layout

- `scripts/<stage>/<corpus>.py`, one file per corpus per stage: `fetch` → (`segment`) → `filter` → `select` →
  `finalize`, plus optional `*_speaker_fixup.py`, `*_prune.py`, and `score/`. `scripts/pipeline/` chains the stages on
  a GPU box.
- `configurations/quality_filtering_<corpus>.yaml`: thresholds, each commented with the reason for its bound.
- `data/` and `tmp/` are gitignored. Downloads go to `data/<Corpus>/`, outputs to `data/processed/<Corpus>/`, and
  listening dumps to `tmp/accepted/` and `tmp/rejected/`.

Run every script from the repo root: default paths are relative, and scripts import `scripts.*` and `tools.*`.
Python 3.12 or later. There are no tests and no linter config.

## Remote GPU boxes

Filtering, segmentation, and codec encoding run on GPU boxes reached over SSH. The box changes often (usually a vast.ai
instance or a local machine), so ask the user which one to use for each task. Run long jobs inside `tmux`, and
`git pull` on the box before you launch. Each user keeps their own SSH aliases and checkout paths in
`CLAUDE.local.md` at the repo root or in the folder above it (gitignored).

## ⚑ Gotchas

- **Use spawn, not fork, once CUDA is initialised.** On Linux, a forked worker makes the multi-speaker metric reject
  every clip silently.
- **A failing metric rejects the clip instead of crashing.** Out-of-memory and CUDA errors show up as
  `<Metric> failed: …` lines and inflated rejection counts.
- **Build CSVs with the `csv` module and the shared dialect.** Rows end in CRLF and transcripts can contain a bare
  `\r`, so `awk`, `cut`, and `paste` split rows.
- **`*_prune.py` deletes audio, codecs, and CSV rows permanently.** Run it with `--dry-run` first.
- **NISQA on MPS returns wrong scores.** Set `nisqa_device: cpu` on a Mac. Only CUDA reproduces existing codec tokens.
- **A dict-valued YAML key replaces its default wholesale,** and a missing `--config` file silently falls back to
  built-in defaults.
- `requirements.txt` misses `faster_whisper` and `omnivoice`, which the segmenter and the ParlaSpeech filter import.
