---
name: add-corpus
description: Bring a new speech corpus into Resonance end to end — inspect the source, write the fetch, filter, select, and finalize stages, calibrate the quality thresholds with listening, and record what was learned. Use when the user names a new dataset or language to add.
---

# Add a corpus

A new corpus takes one script per stage, copied from the closest existing pipeline, plus a YAML of thresholds
calibrated on this corpus, plus a knowledge note. Almost every corpus so far had traps that only showed up on
inspection or calibration (mislabelled columns, wrong durations, broken transcripts, narrators with band-limited
audio), and thresholds borrowed from another corpus have been wrong every time. Budget most of the effort for steps 1
and 4. [metrics.md](metrics.md) explains every cascade stage, its bounds, and how they were calibrated.

## 1. Inspect the source

Download one shard and read the dataset card. Establish, and write down for the note:

- **Format and layout:** parquet, tar with sidecars, or folders of audio; shard naming (hashes, stale duplicates);
  size per shard and in total.
- **Audio:** sample rate (24 kHz is native; 16 kHz sources fail the bandwidth gate), codec, real bandwidth.
- **Transcripts:** curated or raw captions; punctuation; digits and symbols; casing; encoding errors (CML-TTS Polish
  had 14% mojibake).
- **Speakers:** whether `speaker_id` means a person, a book, a source recording, or nothing; gender labels.
- **Columns you can't trust:** check durations, IDs, and quality columns against the audio (CML-TTS durations are 8.8%
  too long; WolneLektury's `mp3` column holds WAV; constant quality scores carry no signal).
- **Licence and access:** gated datasets need the licence accepted and `HF_TOKEN`.

Tell the user what you found, and agree on whether the corpus is worth the work, before writing code.

## 2. Pick the template

| The corpus is | Copy the pipeline of |
|---|---|
| Parquet with 24 kHz audio | WolneLektury or CML-TTS |
| A folder of wav files with transcripts | LibriTTS |
| Tar archives with sidecars, or needs a per-source pass | Emilia |
| Audio that must be regenerated (low sample rate, but good content) | ParlaSpeech-PL |
| Long-form audio that needs sentence segmentation | YODAS2-Sidon (it adds a `segment` stage) |

Not every pipeline has every stage, so copy each stage from the closest corpus that has it. Keep the copied structure
and change only what the corpus needs.

## 3. Write the stages

1. **Fetch** (`scripts/fetch/<corpus>.py`): reuse the `download()` and `token()` helpers the other fetchers import;
   support `--first`, `--last`, and `--output-dir`; skip shards already on disk.
2. **Config** (`configurations/quality_filtering_<corpus>.yaml`): start from the template corpus's YAML. Every key is
   defined in `QualityConfig` (`tools/metrics/types.py`). Leave the bounds as they are until step 4.
3. **Filter** (`scripts/filter/<corpus>.py`): writes `<corpus>_filtered.csv` in the shared CSV dialect (below). Keep
   the template's `--limit`, `--verbose`, dump, and `--workers` flags, since step 4 depends on them. Any worker pool that
   touches CUDA must use `multiprocessing.get_context("spawn")`.
4. **Select** (`scripts/select/<corpus>.py`): calls `tools.prosody.selection.select_bounded`; pass `is_high` from
   gender labels when they exist.
5. **Finalize** (`scripts/finalize/<corpus>.py`): writes `data/processed/<Corpus>/audio/` and `codecs/<name>.npy`
   (int16 `[8, T]`, Higgs tokens through `tools/codec/higgs.py`).
6. **Speaker fixup and prune** (`*_speaker_fixup.py`, `*_prune.py`): only when speaker labels are missing or are
   pseudo-labels.

**CSV dialect:** `|` delimiter, `quoting=csv.QUOTE_NONE`, `quotechar=None`, `escapechar="\\"`, columns
`dataset|name|transcription|speaker_id|language` (plus `speaker_gender` when known). Rows end in CRLF and transcripts
can hold a bare `\r`, so build and merge CSVs with the `csv` module only.

## 4. Calibrate the filter

Follow the calibration loop in [metrics.md](metrics.md):

1. Run the filter on a random sample with `--limit 150 --verbose` and the dump flag.
2. Compare the verdict breakdown with the known corpora; investigate any stage far above its usual rate before
   touching its bound.
3. For NISQA, compare the score percentiles with a clean reference corpus instead of reusing absolute bounds.
4. Ask the user to listen to `tmp/accepted/` and `tmp/rejected/`; listening decides.
5. Set per-corpus bounds with a reason in a comment on every changed key, and re-run on a fresh sample (300–500
   clips) until the user accepts the result.

Expect to keep roughly a third to a half of a web corpus, and more of a studio one.

## 5. Run it and check the output

Run the full pipeline with the `run-pipeline` skill. Before calling the corpus done, check the output:

- `data/processed/<Corpus>/` has one `.npy` per CSV row, and the CSV row count matches the selection.
- Hours, speakers, and minutes per speaker look like the calibration predicted.
- Codecs were encoded on CUDA (other devices change 3–10% of tokens).

## 6. Record it

Write `knowledge/<corpus>.md` following the note style in `.claude/rules/general.md`:

- layout and column traps (a `| Column | Trap |` table);
- the first calibration run as a verdict table, with sample size, config, and throughput;
- each recalibrated bound with its score percentiles and the listening result;
- yield (accepted share, hours, speakers) and cost (time and disk per shard);
- `## ⚑ Gotchas` and `## Open points`.

Add its line to `knowledge/INDEX.md`.

## Done

The corpus is done when its selected, finalized subset is in `data/processed/<Corpus>/`, the output checks in step 5
pass, the user has accepted a listening sample, and the note and its index line are written.
