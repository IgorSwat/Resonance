# The quality cascade: metrics, bounds, and calibration

Reference for choosing and checking thresholds. The design and the full calibration history are in
`knowledge/data_filtering.md` and the corpus notes; numbers here cite them.

## How the cascade runs

`tools/metrics/pipeline.py` runs, in order: length gate (`min_duration`, `max_duration`), effective bandwidth,
clipping, mains hum, spectral flatness, multi-speaker (if enabled), NISQA, CTC alignment (if enabled). A clip stops at
the first stage that rejects it, so later stages never see it. Multi-speaker runs before NISQA on purpose: it's
cheaper and rejects more.

- **An exception inside a metric rejects the clip** and prints `<Metric> failed: …`. Out-of-memory errors, CUDA
  errors, and cuFFT failures therefore show up as rejections, not crashes. Read the log for these lines.
- **Config:** `QualityConfig.from_yaml` (`tools/metrics/types.py`) raises on unknown keys. A missing `--config` file
  silently falls back to the dataclass defaults (the script prints "built-in defaults"), and those differ from every
  YAML (`min_duration` 5.0 instead of 3.0, NISQA gating MOS and noisiness). A dict-valued key (`nisqa_min`,
  `ctc_max`, `multi_speaker_max`) replaces its default entirely.
- Filter-level passes (source rejection, speaker drift, digit rejection, output loudness, verbalization, duplicate
  sources) run only in the filter scripts that implement them; the cascade ignores those keys.

## Stages

| Stage | Catches | Misses | Cost |
|---|---|---|---|
| Effective bandwidth | artificial band limits: upsampled 16 kHz audio, lossy-codec brickwalls, telephone band | high frequencies synthesised by speech restoration (Sidon) | sub-ms, CPU |
| Clipping | runs of ≥ 3 samples above \|0.99\| | soft clipping, limiter pumping | ~0 |
| Mains hum | a 50/60 Hz line locked within 0.3 Hz of the grid | off-grid buzz, hum masked by speech | ~0; needs ≥ 1.37 s |
| Spectral flatness | tonal noise floors (hum, music beds) and hiss, only when the floor is audible | — | ~0 |
| Multi-speaker | a second voice or overlap, via `pyannote/segmentation-3.0` | clean turn-taking between similar voices | ~10–45 ms per clip |
| NISQA | dropouts and glitches (discontinuity), coloration | content-driven MOS swings | ~6 ms per clip on GPU |
| CTC alignment | missing words, extra or trailing speech, transcript mismatches | 1–2-word errors, mispronunciation | ~22 ms per clip on GPU |
| Speaker drift (filter-level) | a voice change within a clip, via ECAPA | overlapped speech | ~11 ms per clip |

### Effective bandwidth

- **Score:** the share of Nyquist still carrying speech dynamics; it walks up from 3 kHz and stops at a cliff (dynamics
  below 0.4× the speech bands, or a level drop over `bandwidth_cliff_db`, 20 dB).
- **Bound:** `bandwidth_min_frac` 0.85. With 1 kHz bands at 24 kHz the score moves in steps of 1/12, so 0.85 means
  content up to about 11 kHz (10/12 = 0.833 fails). Use `bandwidth_band_hz: 500` when you need finer resolution.
- **Calibration:** keeps 96% of LibriTTS and 100% of LJSpeech; flags every labelled brickwall speaker; monotone on
  synthetic lowpasses (`knowledge/data_filtering.md`).
- **Trap:** bandwidth is often a property of a narrator or chapter, not a clip. CML-TTS Polish lost 47% here, by whole
  narrator. Check the per-speaker pattern before you loosen the bound; a real defect looks like whole speakers
  failing.

### Clipping, hum, flatness

- **Clipping:** `clipping_max_ratio` 1e-4, a rule of thumb, never rejected anything on WolneLektury or YODAS.
- **Hum:** `hum_max_prominence` 15 dB. The one confirmed hum file scored 23 dB and everything else ≤ 13.3 dB; the old
  6 dB bound flagged 42% of clean corpora. Hum is often per speaker (LibriTTS speaker 3526 hums in every chapter).
- **Flatness:** reject below `flatness_min` 0.0005 or above `flatness_max` 0.25, but only when the gap between speech
  and the noise floor is under 35 dB (an inaudible floor always passes). The scale tops out at 0.56 (white noise), not
  1. On web audio the upper bound behaves inversely (high flatness went with high MOS on YODAS); clips survive through
  the gap escape. The note argued for a lower bound of 0.005; the code and YAMLs use 0.0005, reason unrecorded.

### Multi-speaker

- **Scores:** `second` (seconds of the second-most-active speaker in any window) and `overlap` (seconds with two or
  more speakers active). Bounds `{second: 0.0, overlap: 0.0}`: recall first. Raise `second` to about 0.5 for a
  precision-first gate.
- **Calibration** (`knowledge/emilia.md` §2): by ear, 30% of 20 blind clips had a second speaker, against 36.5%
  flagged. `overlap > 0` had precision 13/13; every confirmed positive scored `second ≥ 0.14`. The score is bimodal,
  so the exact cut barely matters.
- **Uncalibrated below 6 s;** flag rates fall with duration (16.7% at 3–4 s vs. 35.3% at 6–10 s).
- **Misses** clean turn-taking: about 17% of gate-passed clips in a conversational pool still had a second voice. The
  source pass covers this.
- **Source pass** (Emilia: per speaker; YODAS: per video): drop the whole source when the share of its clips flagged
  at the multi-speaker stage reaches `source_max_flag_rate`, with at least `source_min_clips` scored. On Emilia, the
  speaker's flag rate predicted a second voice at AUC 0.900 vs. 0.622 for the clip's own score. Costs: 31.7% removed
  per-clip only, 39.9% at 0.5, 50.4% at 0.3. Emilia's YAML uses 0.33, YODAS 0.3.

### NISQA

- **Dimensions** in output order: mos, noisiness, discontinuity, coloration, loudness. Gate mainly on
  **discontinuity** (dropouts, the most stable) and **coloration**. MOS and noisiness move with content, so gating on
  them rejects good clips.
- **Default bounds** `{discontinuity 3.75, coloration 3.5}` sit at LJSpeech's minima.
- **MPS gives wrong scores** (MOS off by 0.64 on average; 57% pass on MPS vs. 100% on CPU for the same clips). Set
  `nisqa_device: cpu` on a Mac.
- **Absolute bounds don't transfer between corpora.** Emilia's bounds rejected 37% of WolneLektury, a clean audiobook
  corpus; `discontinuity ≥ 4.0` alone rejected 27.8%. Compare the new corpus's p5 / p50 / p95 per dimension with a
  clean reference corpus (the table in `knowledge/wolne_lektury.md` §5) and set bounds that cut outliers, not a fixed
  share of a clean corpus.

### CTC alignment

- **Scores** (MMS forced alignment of the given transcript, with uroman for non-Latin text): `loss` per token,
  `maxrate` (max characters per second over 4–8-word windows), `gap_speech` (non-blank posterior between aligned
  words), `trail_speech` (after the last word).
- **Rejects** when `loss > max_loss`, or `loss > loss` bound and `maxrate > maxrate` bound (missing words); or
  `gap_speech > 0.06` or `trail_speech > 0.05` (extra speech).
- **Calibration** (LibriTTS, 6,000 clips): loss median 0.116, p95 0.80, p99 2.11. The combined rule flagged 4 clips,
  all confirmed (100% precision). Missing 1–2 words is nearly invisible to it.
- **Traps:** MMS can't read spoken numerals, so digits or number words in transcripts inflate the loss (YODAS: 56% of
  clips with digits failed vs. 13% without numbers). Clip boundaries that cut into speech fail as missing words.
  Configs set `max_loss` 1.0–2.0; the code default is 5.0.

### Speaker drift

- **Score:** the minimum pairwise ECAPA cosine between 2 s windows. Flag by z ≤ `speaker_drift_max_z` (−2.0) within
  duration bands (0–6, 6–9, 9–13, 13–18, 18+ s), because the raw score falls with duration.
- **Validation:** of the 10 lowest-z clips out of 500, 7 were multi-speaker by ear. Recall is unknown.

## Calibration loop

1. **Sample.** Run the filter with `--limit N --verbose` and the corpus's dump flag on a random sample: 100–150 clips
   for a first look, 300–500 before you commit to bounds. On YODAS, `--limit` counts recordings, not clips.
2. **Read the breakdown** the filter prints at the end: count and share per verdict, accepted hours, speakers,
   minutes per speaker, and throughput. In `--verbose` mode each clip prints `REJECTED by <stage>` and every bounded
   score against its allowed range.
3. **Compare with known corpora** (first runs, sample of 100–150):

   | Verdict | WolneLektury | CML-TTS PL | CML-TTS ES | YODAS2-Sidon |
   |---|---|---|---|---|
   | accepted | 48% | 34% | 34% | 35–52% |
   | effective_bandwidth | 0 | 40% | 11% | 0–4% |
   | nisqa | 37% | 1% | 28% | 2–18% |
   | ctc_alignment | 6% | 5% | 25% | 31–34% |
   | multi_speaker | 6% | 3% | — | 11–16% |
   | too_short | 3% | 16% | 0 | 0–77% (segmentation) |

4. **Investigate any stage far above these rates** before changing its bound (see the red flags below), using
   percentiles of its scores and the dumped clips.
5. **Listen.** Ask the user to listen to `tmp/accepted/` and `tmp/rejected/`. The dump files list each clip's
   verdict and scores (`rejections.txt`, `nisqa.txt`, or `scores.txt`, depending on the filter). Listening decides
   whether a rejection is right.
6. **Set per-corpus bounds** in the YAML, with a comment on every changed key giving the reason and the measurement.
   Re-run on a fresh sample until the user accepts it.

## Red flags

- **multi_speaker near 100%,** with `MultiSpeakerMetric failed: CUDA driver initialization failed`: a worker pool
  forked after CUDA was initialised. Use a spawn context.
- **NISQA rejecting 35–45% of clean studio audio:** borrowed absolute bounds, or NISQA running on MPS.
- **Rejections that rise as you add workers:** out of memory (YODAS: 10 workers lost clips silently; 6 is the limit
  on a 24 GB card).
- **A scatter of NISQA failures with cuFFT errors:** too many workers on one GPU.
- **effective_bandwidth near half:** check whether whole speakers fail; that's a real defect.
- **ctc_alignment high:** transcripts that don't match the audio end to end, clip boundaries cutting speech, or digits
  in transcripts.
- **too_short dominant:** the segmentation is wrong upstream.
- **`EffectiveBandwidthMetric failed:` with "Silent or constant-level audio":** digitally silent clips.
- **"built-in defaults" in the run header:** the `--config` path is wrong.
