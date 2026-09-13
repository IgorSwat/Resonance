# WolneLektury — layout, column traps, first filter run

Measured while building the WolneLektury filtering path (`scripts/fetch/wolne_lektury.py`,
`scripts/filter/wolne_lektury.py`, `scripts/finalize/wolne_lektury.py`). Numbers come from
`data/WolneLektury/train-00000-of-00380.parquet` (1,000 rows) unless stated.

Companion note: `knowledge/data_filtering.md` for the quality cascade itself.

---

## 1. The `mp3` column is not mp3

`datadriven-company/WolneLektury-TTS-Polish` stores each clip in a column named `mp3`, but the
bytes are **24 kHz mono WAV PCM_16** (`RIFF` magic, `mp3.path` ends in `.wav`). Verified on
every clip sampled.

Why it matters: 24 kHz is already the Higgs codec's rate, and effective bandwidth measured
**1.00 of Nyquist on 20 of 20** clips, so the audio is genuinely full-band rather than upsampled.
There is therefore no reason to regenerate this corpus through OmniVoice the way
`scripts/filter/parla_speech_pl.py` regenerates its 16 kHz source. The filter judges the
recordings as they are and writes the accepted ones out as wav.

## 2. Two shard naming schemes on HuggingFace — one is stale

`data/` in the repo holds **385** files under two names:

- `train-XXXXX-of-00381.parquet` — ids 0–380, contiguous, **the real corpus** (383,710 samples,
  997 h, matching the dataset card).
- `train-0000{0,1,2,3}-of-00380.parquet` — four leftovers from an earlier upload.

They are not the same content: shard 0 is 432,143,351 bytes under `-00380` and 432,954,809 bytes
under `-00381`. `scripts/fetch/wolne_lektury.py` fetches the 381 set only.

## 3. Column traps

| Column | Trap |
|---|---|
| `__key__` | The clip id; 12 hex chars, no speaker or book prefix in it. |
| `mp3` | WAV, not mp3 — see §1. |
| `speaker_id` | Per the dataset card this is an **audiobook identifier**, not a narrator (e.g. `grabinski-swiadek-materna`). The card claims 1,207 narrators; shard 0 carries 20 `speaker_id` values over 1,000 rows. A speaker-level gate acts on a recording, not necessarily a person. |
| `gender` | Values are `male`/`female`, **not** the `M`/`F` that `scripts/select/parla_speech_pl.py` expects in `HIGH`. The filter writes them through unchanged into `speaker_gender`. |
| `dnsmos` | **Constant per audiobook** — a single value, 3.5, across all 1,000 rows of shard 0. Useless as a per-clip quality signal; NISQA does that work instead. |

## 4. First calibration run

100 random clips from shard 0, full cascade, defaults of
`configurations/quality_filtering_wolne_lektury.yaml`:

| Verdict | Clips |
|---|---|
| accepted | 48 |
| nisqa | 37 |
| ctc_alignment | 6 |
| multi_speaker | 6 |
| too_short | 3 |

Throughput was 5.1 clips/s single-process on CPU. **No clip was rejected on bandwidth, clipping,
hum or flatness** — the DSP stages have nothing to say about this corpus, and NISQA carries
essentially the whole rejection.

## 5. The borrowed `nisqa_min` bounds are miscalibrated

The bounds in §4 were copied from the Emilia config. Scored through `tools/metrics/nisqa.py`,
they reject **42% of LibriTTS against 37% of WolneLektury** (97 WolneLektury clips, 126 LibriTTS
clips), so they are not detecting anything specific to this corpus — they are a strict absolute
gate that a clean 24 kHz audiobook corpus fails just as often.

`discontinuity: 4.0` is the single worst offender: it alone rejects 27.8% of WolneLektury. The
repo's own `DEFAULT_LBOUND` is 3.75, and `knowledge/data_filtering.md` advises percentile gates
per corpus rather than shared absolutes.

| dimension | bound used | WolneLektury p5/50/95 | LibriTTS p5/50/95 |
|---|---|---|---|
| mos | 3.15 | 2.61 / 3.81 / 4.64 | 2.74 / 4.00 / 4.85 |
| noisiness | 3.40 | 3.16 / 3.88 / 4.31 | 2.39 / 3.75 / 4.49 |
| discontinuity | 4.00 | 3.54 / 4.18 / 4.47 | 3.90 / 4.42 / 4.74 |
| coloration | 3.50 | 3.50 / 4.07 / 4.36 | 3.47 / 4.27 / 4.55 |
| loudness | 3.00 | 3.25 / 4.04 / 4.43 | 3.40 / 4.30 / 4.69 |

### NISQA itself is not the problem

Two hypotheses were tested and ruled out:

- **Sample rate.** NISQA's mel filterbank runs to `ms_fmax = 20000` and the code never resamples,
  so 5 of its 48 mel bands are identically empty on 24 kHz audio. Rescoring a clip at 48 kHz
  moves every dimension by less than 0.11, so this costs nothing in practice.
- **Polish sibilance.** WolneLektury clips carry more 10–11.5 kHz energy than LibriTTS, but the
  rise is present in the **noise floor during silence** (median −1.1 dB against LibriTTS's
  −4.5 dB for 10–11.5 kHz minus 7–8.5 kHz), where no sibilant can contribute. It is a processing
  artifact of the corpus, not phonetics, and `discontinuity` tracks it at r = −0.47.

So NISQA's ranking is meaningful; only the absolute cut points are wrong. A clip rejected here is
usually a genuine outlier: `c376e414c8b8` sits at p1–p4 on all five dimensions with a +4.7 dB
noise-floor rise, against a median of +3.4 dB for the rest of its own audiobook.

## 6. Prosody selection costs two scans of the shards

`tools/prosody` needs each clip twice: once for its pitch median, and again for a contour tracked
inside its speaker's adapted range, which is not known until every median for that speaker is in.
The file-based selectors just open the clip twice. A parquet row has no path, so
`scripts/select/wolne_lektury.py` scans `--root` once per pass instead, plus a third partial scan
when `--dump-accepted` is set.

Budget for that on a full run: 381 shards at ~430 MB is ~164 GB read per pass. Memory stays flat
either way, since only a float per clip survives the first pass.
