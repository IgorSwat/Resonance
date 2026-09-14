# CML-TTS — layout, column traps, per-narrator bandwidth

Measured while building the CML-TTS path (`scripts/fetch/cml_tts.py`, `scripts/filter/cml_tts.py`,
`scripts/finalize/cml_tts.py`). Numbers come from the Polish `dev` shard of `ylacombe/cml-tts`
(853 rows, 4 narrators) unless stated.

Companion note: `knowledge/data_filtering.md` for the quality cascade itself.

---

## 1. Layout

One parquet directory per language (`polish/`, `german/`, …), each split into `train`, `dev` and
`test`. Polish holds 12 train shards plus one dev and one test, ~8 GB in total.

**Shard names carry a content hash** — `train-00000-of-00012-e5cf3a9ceb8d8a0e.parquet` — so they
cannot be derived from the index the way ParlaSpeech and WolneLektury names can. The fetcher lists
the repo over the HTTP tree API and selects from that listing.

## 2. Column traps

| Column | Trap |
|---|---|
| `audio` | 24 kHz WAV. Its `path` (`5090_1447_000098.wav`) is the **only per-clip id**; there is no key column. The stem is speaker_chapter_index, MLS style. |
| `duration` | **Wrong by 24000/22050 = 1.0884**, on 40 of 40 clips checked. It is `wav_filesize / 2 / 22050` against audio stored at 24 kHz, so every value reads 8.8% long. Compute duration from the audio. |
| `text` | The real transcript, punctuated and cased — better than MLS, which is neither. But see §3. |
| `transcript_wav2vec` | An ASR reading of the clip, not a transcript. Useful only as a comparison. |
| `levenshtein` | Similarity between the two. Reads 0.13 at the 5th percentile: a long tail of clips whose text stops well short of what the audio says. The CTC stage catches these. |
| `speaker_id` | An integer, and a real narrator rather than a diarisation label. |
| — | There is no language column; the directory is the language. No gender column either. |

## 3. 14% of Polish transcripts are mojibake

119 of 852 Polish dev transcripts contain letters outside the Polish alphabet, overwhelmingly
`š` where `ą` belongs and `œ` where `ś` belongs (88 and 70 occurrences), plus stray `ÿ` and `é`.
`transcript_wav2vec` has the correct letters in the same rows, so the damage is in the text
pipeline, not the source.

Repair is **not** a clean round trip. `text.encode('iso8859-2').decode('cp1250')` fixes the
`š` → `ą` class exactly, but `œ` (U+0153) has no ISO-8859-2 encoding, so at least two different
mis-decodes are mixed in one field. Treat a gate on non-Polish letters as the cheap option and
repair as a project.

## 4. Bandwidth varies by narrator, and that is the dominant filter

The corpus is LibriVox, so each narrator's recording chain and mp3 bitrate is their own. CML-TTS
upsampled everything into a 24 kHz container, but the content bandwidth did not follow.

| speaker_id | clips | median frac of Nyquist | implied lowpass |
|---|---|---|---|
| 5090 | 44 | 1.00 | none |
| 11295 | 73 | 1.00 | none |
| 7775 | 46 | 0.83 | ~10 kHz |
| 8502 | 61 | 0.58 | ~7 kHz |

47% of clips fall below the 0.85 `bandwidth_min_frac` gate, and they fall as whole narrators
rather than at random. **So "CML-TTS is 24 kHz" is true of the container only.** Budget for
roughly half the advertised hours surviving, and expect the survivors to be a subset of the
narrators rather than a thinning of all of them.

## 5. First calibration run

150 random clips from the Polish dev shard, full cascade, defaults of
`configurations/quality_filtering_cml_tts.yaml`:

| Verdict | Clips |
|---|---|
| accepted | 51 |
| effective_bandwidth | 60 |
| too_short | 24 |
| ctc_alignment | 8 |
| multi_speaker | 5 |
| nisqa | 2 |

8.4 clips/s on 4 workers. Only 2 of the shard's 4 narrators survived, which is §4 showing through.
