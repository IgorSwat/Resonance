# Dataset filtering & selection for TTS training

Notes on filtering in-the-wild speech corpora (Common Voice, MLS/CML-TTS, ParlaSpeech,
YODAS/Granary, audiobooks) down to a training set for Echo. Three axes: **quality**,
**diversity**, **text alignment**.

Organizing principle: **build a cascade, cheapest filter first, and store scores rather
than dropping rows.** Every stage below should become a column in a metadata parquet, not
a delete. That allows re-thresholding without recomputation, and allows training broad →
fine-tuning on the clean slice instead of discarding data.

Two levers Echo already has that most pipelines pay extra for:

- A Mimi encode pass over every file (`scripts/preprocess/codecs.py`) → free reconstruction-based
  quality scoring.
- A CTC forced aligner (commit `82f0fac0`, "forced alignment injected as direct conditioning")
  → free alignment scoring, speaking rate, and pause structure.

---

## 1. Quality — beyond DNSMOS

DNSMOS P.835 OVRL covers one axis (noise-suppression quality) and is blind to several
failure modes that matter for a 24 kHz codec model. Pick *complementary* axes rather than a
second MOS predictor that correlates ~0.8 with the first.

### Tier 0 — pure DSP, ~0.001× realtime, no GPU

| Signal | Catches | Rule of thumb |
|---|---|---|
| **Effective bandwidth** (spectral cutoff + dead-band floor — see [§1.1](#11-effective-bandwidth-in-detail)) | Upsampled 16 kHz content, low-bitrate codec lowpass, telephone-band audio | Drop if cutoff < 85% of Nyquist when targeting 24 kHz Mimi. **Highest-value cheap filter for this setup** |
| **Brickwall detection** (sharp spectral cliff + near-silent band above) | Codec-transcoded audio — sounds fine to DNSMOS but teaches the codec a lie | Dead-band floor below −70 dB = digitally zeroed band |
| **Clipping rate** (fraction of \|x\|>0.99, esp. runs ≥3) | Distortion, over-limited YouTube audio | Drop if > 0.01% of samples |
| **Mains hum** (50/60 Hz + harmonics vs local floor) | Bad recording chains — common in European parliamentary/amateur audio | > 6 dB peak over local floor |
| **Digital dropouts** (runs of exact zeros mid-utterance) | Packet loss, bad segmentation | Any zero-run > 20 ms inside speech |
| **Noise-floor stats** (non-speech frame energy, spectral flatness) | Steady broadband noise; flatness separates hiss from hum/music | P10 vs P90 frame energy → crude SNR |
| **WADA-SNR** | Model-free SNR estimate | < 15 dB risky, < 10 dB drop |
| **LUFS / DC offset / silence ratio** | Normalization problems, mostly-silence segments | R128 target; silence ratio > 50% → re-segment |
| **Voicing/F0 stability** | Whispered, creaky, heavily processed or non-speech content; F0 discontinuities also flag concatenation seams | — |

```python
def clip_rate(y):        return float((np.abs(y) > 0.99).mean())

def zero_run_ms(y, sr):  # longest digital dropout
    z = (np.abs(y) < 1e-5).astype(np.int8)
    d = np.diff(np.flatnonzero(np.diff(np.r_[0, z, 0])))[::2]
    return (d.max() if len(d) else 0) / sr * 1000
```

### 1.1 Effective bandwidth in detail

The most valuable Tier-0 filter for a 24 kHz codec model, and worth spelling out because the
naive implementation does not work.

**What it measures.** Not "does this audio have treble" but "is there an *artificial* band
limit". Natural speech rolls off gradually above ~1 kHz, but fricatives (/s/, /ʃ/, /f/) put
real energy up to 8–12 kHz, and the decay is content-dependent — it moves as phonemes change.
An artificial band limit is a **cliff** (20–60 dB within a few hundred Hz) followed by a
**flat, dead floor** that never moves regardless of what is said. That floor is the signature.

**Causes and signatures:**

| Cause | Signature |
|---|---|
| Source natively 16 kHz, resampled up | Cliff at 7–8 kHz, flat floor above (~−45 dB dither/resampler noise). Very common: MLS, VoxPopuli, LibriSpeech are 16 kHz natively and packagers upsample to hit a target rate |
| Lossy codec (MP3/AAC/Opus, low bitrate) | Sharp brickwall at a *quantized* frequency (11/15/16/19 kHz), near-silent band above (< −70 dB). Common Voice ships MP3 |
| Telephone / VoIP | Cliff at 3.4 kHz (narrowband) or 7 kHz (wideband) |
| Aggressive denoiser / noise gate | Spectral holes, often above ~10 kHz, sometimes time-varying |
| Genuine full-band recording | No cliff; content reaches near Nyquist, floor tracks room noise |

Two files can have identical DNSMOS and identical SNR while one is full-band and the other is
8 kHz content in a 24 kHz container. DNSMOS judges noise-suppression quality, not bandwidth —
a genuine blind spot, not a redundant check.

**Why it matters for Echo.** Mimi is 24 kHz → Nyquist 12 kHz. Band-limited data becomes an
*uncontrolled nuisance variable*: the model sees bright and dull versions of similar speech
with no conditioning signal separating them, so brightness entangles with speaker identity
and gets sampled arbitrarily at inference. The RVQ layers also waste capacity encoding a dead
band.

**Measured on `data/librispeech/audio/` (24 kHz container, Nyquist 12 kHz):**

```
  1k    2k    3k    4k    5k    6k    7k    8k    9k   10k   11k   12k
   2     2   -10   -13   -25   -36   -49   -48   -44   -46   -46   -47   dB
                                   └──────── flat floor, no speech ────────┘
```

Speech hits the noise floor between 6–7 kHz; above that is a flat −45 dB floor to Nyquist —
about half the modelled band carries no speech information. The detector returns 5.5 / 5.7 /
5.5 / 5.6 kHz across files: **agreement to ±0.2 kHz is itself the diagnostic**, since real
recordings never agree that closely. Corpus-wide consistency ⇒ processing artifact.

For contrast: LJSpeech (22.05 kHz) reaches ~9–10 kHz of an 11 kHz Nyquist with a gradual,
content-dependent rolloff and a high floor (−14 to −30 dB) — genuinely full-band. The 44.1 kHz
`output.wav` in the repo root falls −43 → −51 → −66 → −81 dB and sits dead flat at −82 dB from
15 kHz up — textbook lossy brickwall.

**Reproduction note (`tools/metrics/effective_bandwidth.py`, 50 random files/corpus).** The
±0.2 kHz agreement above does not hold over a random sample — the four files it was measured on
were not representative. Measured cutoffs: LibriSpeech 9.08 ± 1.62 kHz (frac 0.757 ± 0.135, range
0.33–0.91), LJSpeech 9.53 ± 0.62 kHz (frac 0.864 ± 0.056, range 0.73–0.93). The corpora still
separate on `frac` (0.76 vs 0.86) and on dead-band floor (−33.8 vs −19.4 dB), but the cutoff is
*more* variable within LibriSpeech than within LJSpeech, so **within-corpus cutoff agreement is
not usable as the diagnostic** — use the floor and the frac distribution instead. At `frac ≥ 0.85`
the gate keeps 28% of LibriSpeech and 66% of LJSpeech.

Three follow-up findings that change how this metric should be read:

1. **`data/librispeech/` is not LibriSpeech.** 100,416 files, 809 speakers, all 24 kHz PCM_16,
   named `{speaker}_{chapter}_{utt}_{seg}.wav` — that is **LibriTTS/LibriTTS-R**, which is
   natively 24 kHz off the original LibriVox MP3s, not upsampled from LibriSpeech's 16 kHz flac.
   The "packagers upsampled 16 kHz" premise does not apply to this corpus.
2. **No corpus-wide 16 kHz upsampling; per-speaker heterogeneity instead.** Over 40 random
   speakers (6 clips each), median cutoff: 30% below 8.5 kHz, 32% at 8.5–9.5 kHz, 38% at or above
   9.5 kHz. Only **~7–10% of speakers** show the strict upsample/brickwall signature (cliff at
   7.4–8.0 kHz, within-speaker sd < 300 Hz, floor < −55 dB, and 9–11 kHz envelope *not*
   co-modulating with the 1–3 kHz envelope: corr −0.46…+0.10 vs +0.31…+0.77 for full-band
   speakers). The rest of the low-cutoff mass is gradual rolloff over a live ~−35 dB floor —
   dull LibriVox source material, not a digital wall. Bandwidth varies by *recording*, and the recording
   unit is the **chapter** (second filename field), not the speaker: speaker 8008 has two
   chapters live to 12 kHz (−50 dB above 9 kHz) and one walled at 7.7 kHz (−94 dB); speaker 6188
   has one chapter at 8.4 kHz (−95 dB) and one at 11.9 kHz. Within a chapter the wall is tight
   (±33–74 Hz). **Aggregate and gate per chapter.** Corpus-wide scan (656 multi-chapter speakers,
   1,678 chapters, 4 clips each, ~40 s total): **18 speakers mix a full-band chapter with one at
   `frac` ≤ 0.85**, of which four are confirmed 8 kHz upsamples — 7868 (ch 110705/110706 at
   7.83 kHz, −98 dB, vs 246932 full-band), 8008 (271817 at 7.71 kHz, −95), 6014 (32886 at
   8.37 kHz, −93), 6188 (73024 at 8.42 kHz, −94). The rest are band-limited over a *live* floor
   (−60…−85 dB) — dull sources, not digital walls. 6120 ch 56179 is a third case: a dead band
   (−95.5 dB) walled at 10.9 kHz, i.e. a lossy codec rather than a resample.
3. **The floor-relative cliff detector above was replaced** (`tools/metrics/effective_bandwidth.py`).
   Thresholding against the file's own >0.9·Nyquist floor makes the score non-monotone in true
   bandwidth: perfectly full-band AM white noise scored `frac` 0.00, pink AM noise 0.37, and the
   same pink noise hard-lowpassed at 8 kHz scored 0.68 — the band-limited signal beat both
   full-band ones. Its practical ceiling on real speech was ≈0.90, so a 0.85 gate sat within one sd
   of the ceiling and rejected 78% of LibriTTS, of which only 6% had a dead top band and 71%
   carried real speech energy in 9–11 kHz.

**Working formulation: per-band temporal dynamic range.** A speech-bearing band's level moves with
what is said; a dead band's does not. Take P95−P20 of each 1 kHz band's per-frame level (dB) over
the loud 60% of frames, and walk up from 3 kHz until a band falls below 0.4× the 1–4 kHz reference
range. Absolute, sample-rate agnostic, and independent of the noise floor. Measured profiles:

```
band upper edge (kHz)     1    2    3    4    5    6    7    8    9   10   11   12
LibriTTS 7190 (16k src) 11.3 20.1 21.0 23.2 26.3 23.9 21.7 20.2  9.9  3.0  2.9  3.3
LibriTTS 2673 (full)    10.8 27.6 24.8 25.3 29.5 34.0 29.4 30.1 26.1 28.9 24.6 23.9
LJSpeech (22.05 kHz)    11.4 26.9 25.3 19.7 23.6 38.0 51.6 42.9 38.7 40.3 41.6
```

Dead bands collapse to ~3 dB; real content stays above 20 dB. Validation: monotone on synthetics
(pink AM noise lowpassed at 4/6/8/10 kHz / none → 0.42 / 0.58 / 0.75 / 0.92 / 1.00), keeps 96% of
LibriTTS and 100% of LJSpeech at a 0.85 gate, and still flags every labelled brickwall speaker
(7190 → 0.67–0.75, 2774 → 0.67–0.92, 8388 → 0.58–0.83). Caveats: cutoff is quantized to the band
width, and a cliff landing exactly on a band edge reads one band high — clip 6188_73024 scores
0.75 (9 kHz) against a true wall at 8.4 kHz because residual energy in 8–8.5 kHz keeps that band
alive; use `band_hz=500` when per-file precision matters.

**Confirmed upsample signature at high resolution** (8192-point FFT). Content dies by 7.9 kHz with
the band above sitting at **−96 dB — the 16-bit PCM quantization floor, i.e. digitally zeroed**,
not the ~−45 dB of resampler dither this doc originally predicted. The transition begins around
7.0–7.5 kHz (anti-alias passband edge ≈0.92·8 kHz). Distinguish from a lossy-codec brickwall,
which lands at a codec-typical frequency instead of 8 kHz — e.g. chapter 8195_117382 walls at
10.79 kHz ±16 Hz across 8 clips.

**Why the naive version fails.** A 99%-cumulative-energy rolloff saturates around 3–4 kHz even
on pristine audio, because most speech energy is below 1 kHz. And unsmoothed per-bin spectra
are noisy enough that isolated bins poke above any fixed threshold — an unsmoothed −40 dB
threshold reported 11.3 kHz for a file whose content genuinely stops at 6 kHz.

Four things make it robust:

1. **Loud frames only** (above the 60th energy percentile) — silence frames contain only noise floor.
2. **95th percentile per bin, not the mean** — HF energy lives in sibilants, which occupy ~10%
   of frames; averaging buries them.
3. **Smooth over ~500 Hz before thresholding** — the fix that stabilizes the estimate.
4. **Threshold against the file's own dead-band floor**, not a fixed dB value.

```python
def band_report(path, nfft=2048, hop=512, smooth_hz=500):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    w = np.hanning(nfft); n = 1 + max(0, len(y)-nfft)//hop
    F = np.stack([np.abs(np.fft.rfft(y[i*hop:i*hop+nfft]*w))**2 for i in range(n)])
    F = F[F.sum(1) > np.percentile(F.sum(1), 60)]          # 1. loud frames
    L = 10*np.log10(np.percentile(F, 95, axis=0) + 1e-20)  # 2. sibilance-aware
    fr = np.fft.rfftfreq(nfft, 1/sr)
    L -= np.median(L[(fr > 300) & (fr < 3000)])            # normalize to speech band
    k = max(3, int(smooth_hz / (sr/nfft)) | 1)
    Ls = np.convolve(L, np.ones(k)/k, mode="same")         # 3. smooth

    top = Ls[fr > 0.9*(sr/2)]
    floor, flat = np.median(top), top.std()                # 4. dead-band floor
    above = np.flatnonzero(Ls[:-k] > max(floor + 8.0, -60.0))
    fc = fr[above[-1]] if len(above) else 0.0
    return dict(cutoff=fc, frac=fc/(sr/2), floor=floor, flatness=flat)
```

Decision rule: `frac > 0.85` → full-band. `floor < −70 dB` → lossy brickwall (that dead a band
is digitally zeroed). `frac < 0.75` with a flat floor → band-limited source. Calibrate per
corpus by plotting the cutoff histogram — expect **discrete clusters**, not a continuum, since
each cluster is one codec or one resampling decision.

**False positives:**

- Short clips without sibilants genuinely lack HF. The per-bin 95th percentile mitigates it;
  also require ≥ 2 s, or aggregate per speaker.
- Dull mics and low-pitched voices roll off gradually *without* a flat floor. The flatness
  test, not the cutoff frequency, separates these from real band limits — cutoff alone is
  insufficient evidence.
- Noise added after upsampling can mask the cliff. Advanced check: correlate the HF band's
  energy envelope against the speech-band envelope over time. Real speech HF co-modulates
  with the speech; added noise does not.

**What to do with the verdict** (dropping is not the only option, and for Polish — where data
is scarcer — probably not the right one):

1. **Drop** below the bandwidth floor for the main training set. Cleanest.
2. **Condition on it.** Bandwidth is a discrete, reliably measurable nuisance factor → make it
   an explicit tag token (DataSpeech/Parler approach), then request full-band at inference.
   Keeps band-limited data useful for prosody and phonetics without entangling timbre.
   **Preferred given the pl/es/it/pt data situation.**
3. **Lowpass everything to a common bandwidth.** Consistent but permanently gives up brightness.
4. Bandwidth extension — avoid; substitutes hallucinated HF for missing HF.

**Cost:** one FFT over ~30 sampled frames per file is enough. Sub-millisecond, CPU-only,
trivially parallel — can fold into the existing decode loop in `scripts/preprocess/codecs.py`.

### Tier 1 — small neural models, ~0.01–0.05× realtime on GPU

- **NISQA** — the important complement to DNSMOS. Outputs four dimensions: noisiness,
  coloration, **discontinuity**, loudness. Discontinuity is exactly the codec-artifact/glitch
  axis that DNSMOS OVRL smears into a single number. Details in [§1.2](#12-nisqa-in-practice).
- **Torchaudio-SQUIM Objective** — non-intrusive PESQ/STOI/SI-SDR estimates in one small
  forward pass; already in the torchaudio dependency. Caveat: trained on DNS Challenge 2020
  data, so generalizes imperfectly to podcast/parliament audio. Use for *ranking within a
  corpus*, not as an absolute gate.
- **Brouhaha** — one pass gives VAD + speech-to-noise ratio + **C50 room acoustics**. Reverb
  is a major axis DNSMOS barely penalizes, and reverberant training data bakes the room into
  the voice. If adding only one neural quality model, make it this one.
- **UTMOS / UTMOSv2** — naturalness MOS. Catches already-processed audio: denoised,
  bandwidth-extended, or itself TTS-generated (increasingly common in YouTube-sourced corpora).
- **Overlapped-speech detection** (pyannote) — two voices in one clip destroys speaker
  conditioning; diarization alone misses short overlaps.
- **Audio tagging** (PANNs/BEATs, tiny) — background music, applause, laughter. Music is the
  top contaminant in YODAS-style data and DNSMOS often rates musical backgrounds as *clean*.

### 1.2 NISQA in practice

**What it is.** A single-ended (no-reference) MOS predictor trained on crowdsourced ratings.
Mel spectrogram (48 bands, 20 ms window / 10 ms hop) → 150 ms segments (40 ms hop) → framewise
CNN (384-d per segment) → 2 tiny Transformer blocks (d=64) → attention pooling → 5 heads.
Local detection, contextual confirmation, severity-weighted aggregation — attention pooling
is why a 200 ms glitch in a 10 s clip still moves the score instead of averaging away. Window
and hop are defined in *seconds*, so it is sample-rate agnostic: no resampling needed.

**Which outputs to use.**

| Dimension | Responds to | Use it? |
|---|---|---|
| **Discontinuity** | packet loss, dropouts, glitches, clipping | **Yes — the reason to run NISQA.** Most specific and most stable |
| **Coloration** | linear spectral distortion, bandwidth limiting | Secondary; see caveat below |
| Noisiness | additive background noise | Overlaps DNSMOS; content-sensitive |
| Loudness | level | Non-orthogonal to the rest; low value |
| MOS | overall transmission quality | **Skip** — largely redundant with DNSMOS |

**Measured cost** (M-series CPU, 14 threads, torchmetrics NISQA v2.0):

- 275× realtime sequential, **372× realtime batched (batch 8; 32 gives nothing more)**
- ~10 ms fixed overhead + ~2.4 ms per audio-second → short clips are overhead-bound, batch them
- **MPS gives no speedup** (21.7 vs 21.2 ms) — model too small; stay on CPU, parallelize processes
- ⇒ 1,000 h ≈ 3 compute-hours single-process, ~25 min across 8 workers. At this speed audio
  *decoding* is a comparable cost, especially for MP3 corpora.

**Hard input bounds** (identical at 16/24/48 kHz):

- min ~150 ms → `RuntimeError: Input signal is too short.`
- max ~50–55 s → `RuntimeError: Maximum number of mel spectrogram windows exceeded.`
  **Chunk long-form audio** — this bites on ParlaSpeech and YODAS2, which ship unsegmented.

**Multilingual: yes, valid for pl/es/it/pt.** Training data is English-only (Librivox-DNS, TSP,
UK/Ireland dialect, AusTalk), but there is no lexical or phonetic modelling in the architecture
— it learns channel degradations, not language. The authors validated on two German test sets.
Measured on MInDS-14 (protocol-matched, 45 clips/language, 8 kHz):

- every per-language offset vs English is **≤ 0.19** on all five outputs
- **en-US vs en-GB differ by 0.25 on Noisiness — more than any cross-language offset**
- within-language sd is 0.24–0.81, i.e. **2–4× larger than any language-level offset**
- degradation responses transfer cleanly: dropouts → Discontinuity −1.24…−1.49 in all six
  language subsets; noise → Noisiness −1.63…−2.01

Caveat on that test: 8 kHz telephony, so invariance is shown in a compressed range, not at
24 kHz full-band. Re-run the degradation sweep on ~50 clean 24 kHz clips per language once the
corpora are pulled (about a minute per language) to confirm at the real operating point.

**Two caveats that change how you gate:**

1. **It under-reacts to bandwidth limiting.** A 4 kHz lowpass moves Coloration only −0.45 and
   MOS −0.39 — it was trained on telephony, where band limiting is normal rather than a defect.
   Band-limited LibriSpeech even scores *higher* MOS (4.63) than full-band LJSpeech (4.41).
   **Keep the spectral detector of [§1.1](#11-effective-bandwidth-in-detail) as the authority
   on bandwidth; do not substitute Coloration for it.**
2. **Content alone moves the score.** Same speaker, same session, different sentences: MOS
   sd 0.28 (range 3.58–4.79), Noisiness sd 0.31. Discontinuity (0.09) and Coloration (0.10) are
   ~3× more stable — another reason to gate on those. Trust NISQA on distributions, never on a
   single clip.

**Code.** torchmetrics ships the v2.0 weights (needs `librosa` + `requests`; downloads to
`~/.torchmetrics/NISQA/` on first call).

```python
import torch
from torchmetrics.functional.audio.nisqa import (
    non_intrusive_speech_quality_assessment as nisqa,
)

# GOTCHA: order is MOS, Noisiness, DISCONTINUITY, COLORATION, Loudness —
# Discontinuity BEFORE Coloration, not the order the paper lists them in.
MOS, NOI, DIS, COL, LOUD = range(5)

def score_clip(wav: torch.Tensor, sr: int) -> dict:
    """wav: 1-D float tensor. Chunk to <50 s and pad/skip <150 ms before calling."""
    s = nisqa(wav, sr)
    return {"nisqa_dis": s[DIS].item(), "nisqa_col": s[COL].item(),
            "nisqa_noi": s[NOI].item(), "nisqa_mos": s[MOS].item()}

# Batched: stack equal-length clips -> (B, T); returns (B, 5). Batch 8 is the sweet spot.
scores = nisqa(torch.stack(clips_5s), sr)          # functional = per-sample
```

Use the **functional** interface — the module interface (`NonIntrusiveSpeechQualityAssessment`)
reduces across the batch, which is never what dataset filtering wants. For bulk runs the
reference repo's `run_predict.py` (PyPI `nisqa`) takes a CSV of paths and writes a table.

Gate with per-language, per-corpus percentiles rather than shared absolute thresholds — the
language offsets are small but free to remove, and per-corpus channel differences are not.

> Note: NISQA's MOS is *transmission* quality, not naturalness. For judging Echo's own output,
> use the separate **NISQA-TTS** checkpoint (predicts Naturalness) or UTMOS instead.

### Tier 2 — Mimi reconstruction self-consistency (free for Echo)

Every file is already Mimi-encoded. Decode it back in the same pass and score:

- mel-spectrogram L1 / multi-scale STFT distance between original and reconstruction, or
- cheaper: **RVQ residual energy** after the last quantizer, plus entropy / codebook-usage
  histogram of layer 0.

Audio the codec reconstructs poorly is precisely the audio the AR+NAR stack cannot learn,
since everything downstream sees only tokens. This aligns the filter with the *actual* model
bottleneck rather than a generic perceptual proxy, and costs one decode. High residual
energy correlates with noise, reverb, music and out-of-distribution timbre simultaneously.

Natural extension: after a first training run, score utterances by **model loss/perplexity**
under the AR model and drop the top-loss tail. Highest-signal filter available, at the cost
of one throwaway training run.

### Practical notes

- **Threshold per source corpus, not globally.** Common Voice, ParlaSpeech and audiobooks
  have very different DNSMOS baselines. Percentile gates within each source (e.g. drop the
  bottom 20%) avoid deleting an entire corpus by accident.
- Calibration point from Emilia: DNSMOS OVRL > 3.0 plus their other filters kept **38.75%**
  of raw data (258h of 667h), lifting mean DNSMOS from 2.88 → 3.26. Expect to keep a third,
  not 90%.
- Check filter correlation before adding a stage. A new score correlating > 0.85 with an
  existing one buys compute cost, not coverage.

---

## 2. Diversity — better than random

Random selection is a strong baseline but faithfully reproduces the *source* distribution,
which for audiobook/parliament data is mode-collapsed onto neutral narration by a few
prolific readers.

### Level 1 — structural rules (nearly free, most of the win)

- **Per-speaker caps.** Sample at most `k · sqrt(n_speaker)` or `k · log(n_speaker)` per
  speaker instead of proportionally. On e.g. MLS Spanish (918h / 86 speakers) this alone
  transforms the effective speaker distribution.
- **Text deduplication** (MinHash/SimHash on normalized text). LibriVox has the same
  public-domain books read by many volunteers; parliamentary data is full of formulaic
  openers. Dedupe *speaker-aware*: keep duplicates across speakers (good for voice
  diversity), drop them within a speaker.
- **Punctuation/sentence-type balancing.** Nearly free and a strong intonation proxy —
  questions and exclamations are rare in narration but essential at inference. Over-sample
  them deliberately.
- **Duration histogram flattening.** Don't let 3–5 s clips dominate; long-context prosody
  only appears in long clips.

### Level 2 — an explicit diversity feature space

| Group | Features | Cost |
|---|---|---|
| **Speaker** | ECAPA-TDNN / WavLM / ReDimNet x-vector (192–256 d) | small |
| **Prosody** | log-F0 mean, std **in semitones**, range, P5/P95; DCT coeffs 1–4 of the F0 contour (contour *shape*, not just spread); energy variance; spectral tilt / H1–H2 (breathy vs pressed) | free (DSP) |
| **Rhythm** | phones/sec, pause count, pause-length distribution, phone-duration variance | **free** — derive from the existing CTC alignment |
| **Emotion/style** | `emotion2vec_plus` embedding or logits — genuinely multilingual (validated across 10 languages), small model | small |
| **Text** | sentence length, punctuation class, diphone/triphone inventory | free |

The CTC aligner makes the rhythm block free. Speaking rate and pause structure are two of the
strongest prosodic axes, and most pipelines pay for an aligner just to get them.

### Level 3 — selection algorithms

1. **Stratified binning + uniform sampling** over `speaker_cluster × pitch_bin × rate_bin ×
   emotion_bin`. Crude, ~10 lines, captures most of the benefit.
2. **k-means → per-cluster sampling with inverse-density weighting.** Over-samples sparse
   regions. FAISS scales this to millions of utterances.
3. **k-center greedy (max-min distance).** Maximizes worst-case coverage; O(nk) with a FAISS index.
4. **Facility-location submodular maximization** with lazy greedy — the principled
   formulation, (1 − 1/e) ≈ 63% optimality guarantee, standard in the speech data-selection
   literature. `apricot` implements it.
5. **DPP sampling** — elegant but O(n³) without approximation; rarely worth it at scale.

Published result worth knowing: diversity-based core-set selection on linguistic + acoustic
features **beats phoneme-balanced selection**, which beats random, across languages and
corpus sizes.

### Two traps

- **Pure max-diversity selection over-samples outliers**, and outliers in in-the-wild data are
  mostly *defects* (noisy, clipped, mis-transcribed). Run diversity selection *inside* the
  quality-gated pool, and prefer facility-location (rewards representativeness) over pure
  dispersion (rewards extremity).
- **Don't trust the selector — measure the selection.** Report unique-speakers-per-hour,
  prosody-bin histogram entropy, diphone coverage, and question/exclamation fraction before
  and after. If entropy didn't move, the selector isn't doing anything.

### Encoding the "prosodic range over consistency" preference

Define a scalar **expressiveness score** (semitone F0 std + F0 range percentile + energy
variance + rate variance), then sample to *flatten* its histogram — actively upweighting the
expressive tail instead of the narration mode. This states the preference directly rather
than hoping it falls out of a generic diversity metric.

---

## 3. Text-alignment verification without ASR

Worth splitting "non-ASR" into *no acoustic model at all* vs *acoustic model but no decoding*.
The second is where the value is: 50–100× cheaper than a Whisper pass, because there is no
beam search, no language model, and no autoregressive decode.

### Main method: CTC forced-alignment scoring (already available)

Run the CTC model once and score the *given* transcript rather than decoding a new one. This
is what YODAS/Granary does — a CTC loss threshold (they used 2.0) as the alignment gate. One
forward pass yields:

| Derived signal | Failure mode caught |
|---|---|
| **Normalized CTC loss / mean per-token posterior** | Global mismatch — wrong transcript entirely |
| **Leading/trailing unaligned time** | Audio extends past the transcript (extra speech, music intro, applause) |
| **Max intra-path gap** | Missing text mid-utterance — audio has words the transcript doesn't |
| **Per-phone duration outliers vs an IQR prior** | Emilia's filter: anomalous char/phone durations via interquartile-range outlier detection |
| **Blank-posterior ratio** | Low-confidence / non-speech regions |
| **Path monotonicity + confidence entropy** | Repeated or reordered text |

```python
# One forward pass, no decoding — score the transcript you were given.
logp = ctc_model(audio).log_softmax(-1)           # (T, V)
loss = ctc_loss(logp, target, ...) / len(target)  # normalized → thresholdable
path = torchaudio.functional.forced_align(logp, target)
starts, ends, scores = spans(path)

flags = {
    "lead_gap":    starts[0],
    "trail_gap":   audio_dur - ends[-1],
    "max_gap":     (starts[1:] - ends[:-1]).max(),
    "mean_conf":   scores.mean(),
    "dur_outlier": iqr_outlier_frac(ends - starts, phone_prior),
}
```

### Free pre-filters (no model at all)

Run before the CTC pass to kill obvious junk at zero cost:

- **Characters/sec and phonemes/sec vs a language-specific IQR.** Polish, Spanish, Italian and
  Portuguese have different syllable rates — calibrate per language. Catches a 2 s clip with a
  40-word transcript.
- **VAD speech duration vs predicted duration from text length** (phoneme count × mean phone
  duration). Same idea, robust to leading/trailing silence.
- **Speech-region count vs punctuation-group count.** A three-sentence transcript should not
  align to one unbroken 12 s utterance.
- **Text-side sanity regexes** — underrated, catches a *systematic* error class. Unexpanded
  numerals, currency, abbreviations and dates ("2023", "€50", "dr.") mean the audio says
  something the text doesn't spell. Normalize or drop. Also: stray markup, wrong alphabet,
  non-target-language characters.

### Cross-modal semantic check (no ASR, genuinely different signal)

**SONAR** encodes speech and text into the same 1024-d sentence embedding space, with speech
encoders covering pl/es/it/pt. Cosine similarity between the speech and text embeddings is a
semantic match score — one forward pass each, no decoding.

Complementary to CTC in an important way: CTC scores *phonetic* agreement and can be fooled
into a mediocre-but-passing score by a plausible near-miss, while SONAR catches "this
transcript is a different sentence about a different topic". Use CTC as the primary gate and
SONAR on the ambiguous middle band. (Omnilingual SONAR, 2026, extends language coverage.)

### Two more cheap consistency checks

- **Audio LID vs text LID.** A tiny language-ID model on the audio compared against the
  transcript's detected language. Catches code-switching and wrong-language rows, rampant in
  YouTube-derived data, which CTC scores badly but ambiguously.
- **Two-aligner agreement.** Score with two cheap CTC models (e.g. an MMS head and an XLS-R
  fine-tune); disagreement flags rows for review. Cheaper than a Whisper pass and better
  calibrated than either model alone.

---

## Suggested cascade

| Stage | Cost (× realtime) | Keeps |
|---|---|---|
| 1. Text sanity regex + duration-ratio pre-filter | ~0 | ~90% |
| 2. DSP tier — bandwidth, clipping, dropouts, hum, WADA-SNR | ~0.001 | ~75% |
| 3. CTC alignment score + derived gap/duration flags | ~0.01 | ~55% |
| 4. Brouhaha (VAD/SNR/C50) + NISQA + overlap/music tagging | ~0.05 | ~40% |
| 5. Mimi reconstruction consistency (piggybacks the existing encode) | ~0 marginal | ~35% |
| 6. Diversity selection *within* the survivors | minutes total | target budget |
| 7. (optional) loss-based re-filter after a first training run | 1 training run | top slice |

The 35% is a starting threshold, not a truth. Being able to re-cut at 50% or 20% without
recomputing is worth more than getting the threshold right the first time.

---

## References

- [Emilia / Emilia-Pipe](https://arxiv.org/html/2501.15907v2)
- [Granary](https://arxiv.org/html/2505.13404v1)
- [YODAS](https://arxiv.org/html/2406.00899v1)
- [Torchaudio-SQUIM](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html)
- [Brouhaha (VAD / SNR / C50)](https://arxiv.org/abs/2210.13248)
- [emotion2vec](https://arxiv.org/abs/2312.15185)
- [SONAR](https://github.com/facebookresearch/SONAR) · [Omnilingual SONAR](https://arxiv.org/html/2603.16606)
- [Diversity-based core-set selection for TTS](https://arxiv.org/abs/2309.08127)
- [Submodular subset selection for speech](https://ieeexplore.ieee.org/document/6854213/)
- [Coreset selection survey](https://arxiv.org/html/2505.17799v1)
