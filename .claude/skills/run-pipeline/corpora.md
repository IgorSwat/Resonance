# Per-corpus runbooks

Command sequences for every corpus Resonance already supports. Run from the repo root. Filter and select scripts
default `--output` to a bare file name in the working directory, so pass `--output data/processed/<Corpus>/...`
explicitly. Check each script's `--help` before a run; flags change.

## YODAS2-Sidon

Long-form, Sidon-restored YouTube audio, one FLAC per video, no speaker labels. Stages: fetch → segment (VAD,
Whisper, sentence packing) → filter → speaker fixup → (ΔCE scoring) → select → finalize. One script chains all but
finalize:

```bash
scripts/pipeline/yodas2_sidon.sh <lang> <shard> <first> <last> [score [share]]
# for example: pl 000 0 38        en 001 0 49 score        es 100 25 70 score 0.2
```

- **Environment:** `RESONANCE` (default `/workspace/Resonance`), `ECHO` (`/workspace/Echo`), `PYTHON`
  (`/venv/main/bin/python`), `RUNS` (`/workspace`). The run folder is `$RUNS/yodas_<lang><shard>_<first>-<last>/`
  with `run.log` and one log per step.
- **What it does:** a background fetcher keeps at most 3 raw batches on disk. Per batch it segments, deletes the raw
  batch, then filters. After all batches it concatenates the filtered CSVs and runs the speaker fixup in place.
  Without `score` it stops there (`all.done`); run `scripts/select/yodas2_sidon_dce.py --random --share 1.0` yourself.
  With `score` it splits into 5,000-row chunks, scores each from the Echo checkout with
  `scripts/score/yodas2_sidon_dce.py`, merges, and selects the top `share` (default 0.33) by ΔCE.
- **Finalize yourself:** `python scripts/finalize/yodas2_sidon.py --input <selected.csv>`.
- **Resume:** rerun the same command; steps with a marker (`<batch>.fetched`, `.segmented`, `.filtered`, `fixup.done`,
  `score.done`, `chunks/part_NNN.scored.csv`) are skipped. A `<batch>.fetch_failed` marker aborts the run; delete it
  before rerunning. The raw batch is deleted before `.segmented` is written, so a crash there means refetching.
- **Score mode needs an Echo checkout** at `$ECHO` and model configs the script names. It currently names
  `echo/configs/p2_final.json` and `echo/configs/teacher_standard.json`, which no longer exist in Echo; score mode
  fails until those paths are fixed. Tell the user rather than editing it unasked.
- **Throughput on an RTX 4090:** fetch about 30 s per 3 GB batch; segment about 249× realtime; filter about 48 clips/s
  with 6 workers (10 workers run out of memory and reject clips silently); score about 12 clips/s. An English batch
  (`en000/train-00000`) yields about 7,300 utterances (19.7 h), of which the filter keeps about 48%.
- **Disk:** a raw batch is about 3 GB; segments stay in `data/processed/YODAS2-Sidon/segments/` and are never
  deleted (about 3.4 GB per English batch).
- **Data limits:** Polish has one shard of 38 batches; Spanish 8 shards of 130–140 batches; English 34 shards.

## Emilia and Emilia-YODAS

```bash
python scripts/fetch/emilia.py --lang EN --first 0 --last <n>               # --type yodas for Emilia-YODAS
python scripts/filter/emilia.py --batch en-b000000 --output data/processed/Emilia/emilia_en-b000000_filtered.csv
python scripts/filter/emilia_speaker_fixup.py --input <filtered.csv> --output <speakers.csv>
python scripts/select/emilia.py --input <speakers.csv> --output data/processed/Emilia/emilia_selected.csv \
  --min-per-speaker 3 --target-per-speaker 7
python scripts/finalize/emilia.py
python scripts/finalize/emilia_prune.py --root data/processed/Emilia --dry-run   # then without --dry-run
```

- Gated: accept the Emilia licence on Hugging Face and set `HF_TOKEN`. About 2 GB per batch.
- Without `--batch`, the filter pools all batches so duplicate detection works across them.
- German Emilia-YODAS: `--root data/Emilia-YODAS --config configurations/quality_filtering_emilia_yodas_de.yaml`.
- The prune step deletes the mp3, the codec, and the rows in every CSV under `--root`, permanently.

## ParlaSpeech-PL

```bash
python scripts/fetch/parla_speech_pl.py --first 0 --last <n>                 # 123 shards, ~500 MB each
python scripts/filter/parla_speech_pl.py --batch train-00000-of-00123 \
  --output data/processed/ParlaSpeech-PL/parla_speech_pl_filtered.csv
python scripts/select/parla_speech_pl.py --input <filtered.csv> --output <selected.csv> \
  --min-per-speaker 3 --target-per-speaker 7
python scripts/finalize/parla_speech_pl.py --input <selected.csv>
```

- The source is 16 kHz, so the filter regenerates every clip with OmniVoice (the clip as its own voice reference) and
  gates on CTC only. It needs the `omnivoice` package, runs at about realtime on a GPU, and is single-process.
- Transcripts with digits are rejected (about 19%). Output is levelled to −23 dBFS.

## WolneLektury

```bash
python scripts/fetch/wolne_lektury.py --first 0 --last <n>                   # 381 shards, ~430 MB each, 997 h total
python scripts/filter/wolne_lektury.py --output <filtered.csv>
python scripts/select/wolne_lektury.py --input <filtered.csv> --output <selected.csv> \
  --min-per-speaker 3 --target-per-speaker 7
python scripts/finalize/wolne_lektury.py --input <selected.csv>
```

- `speaker_id` is an audiobook, not a narrator. Selection reads every shard twice (about 164 GB per pass).

## CML-TTS

```bash
python scripts/fetch/cml_tts.py --lang PL --split train --first 0 --last 11  # Polish: 12 shards, ~8 GB
python scripts/filter/cml_tts.py --lang PL --output <filtered.csv>
python scripts/select/cml_tts.py --lang PL --input <filtered.csv> --output <selected.csv>
python scripts/finalize/cml_tts.py --lang PL --input <selected.csv>
```

- Spanish is 203 shards, about 70 GB. Calibrate each language separately; Polish loses many narrators to bandwidth,
  Spanish to NISQA and CTC.

## LibriTTS

```bash
python scripts/filter/libritts.py --config configurations/quality_filtering_libritts.yaml \
  --output data/processed/LibriTTS/libritts_filtered.csv
python scripts/select/libritts.py --input <filtered.csv> --output data/processed/LibriTTS/libritts_selected.csv \
  --min-per-speaker 7 --max-per-speaker 20
python scripts/finalize/libritts.py
```

- No fetcher: put the corpus in `data/LibriTTS/`. Always pass `--config`; the default path doesn't exist and the run
  silently uses built-in defaults. About 175 ms per clip, so the full corpus takes hours.
- The selected CSV has no `language` column.
