# Knowledge index

Line format and folder rules are in `.claude/rules/general.md`.

- `data_filtering.md` — the quality cascade: which metrics, how each was calibrated, diversity selection, text alignment
  without ASR, and the CSV line-ending trap — stale (names Echo paths and the Mimi codec; the pipeline now uses Higgs)
- `emilia.md` — Emilia layout, diarisation leakage and the source-level fix, duplicates, speaker linking, pitch skew —
  current
- `wolne_lektury.md` — WolneLektury layout and column traps, first calibration run, miscalibrated NISQA bounds, the cost
  of prosody selection — current
- `cml_tts.md` — CML-TTS layout, column traps, Polish mojibake, per-narrator bandwidth, Spanish vs Polish — current
- `yodas2_sidon.md` — what YODAS2-Sidon contains, caption problems, Whisper re-transcription, source pass, speaker
  fixup, CUDA-only failures — stale (§8, §12, §13 describe the filter before the segment stage)
