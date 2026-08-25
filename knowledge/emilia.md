# Emilia — layout, diarisation leakage, duplication, speaker linking

Everything measured while building the Emilia filtering path (`scripts/filter/emilia.py`,
`tools/metrics/multi_speaker.py`). Numbers come from `data/Emilia/en-b000000` unless stated;
sample sizes are given because several of them are small.

Companion note: `knowledge/data_filtering.md` for the quality cascade itself.

---

## 1. Layout and identifiers

On HuggingFace (`amphion/Emilia-Dataset`, gated) the corpus is WebDataset tars: 2,360 for
Emilia, 1,983 for Emilia-YODAS. Locally a shard is unpacked to one directory per tar
(`data/Emilia/en-b000000/`), each clip an `.mp3` beside a `.json` sidecar.

**The `B#####` in a clip ID is not the tar index.** The mapping is:

```
HF tar index = source_batch * 10 + (speaker_index mod 10)
```

- `EN-B000000.tar` → batch `B00000`, speakers S00000, S00010, … S09990
- `EN-B000001.tar` → batch `B00000`, speakers S00001, S00011, …
- `EN-B000010.tar` → batch `B00001`; `EN-B000100.tar` → `B00010`; `EN-B001000.tar` → `B00100`

Consequences: **one speaker ID never spans two tars** (its utterances are contiguous and start
at `W000000`), but **one source batch is spread over 10 tars**. Emilia-YODAS is different —
IDs are `EN_<youtube_id>_W######`, clips are shuffled, and one video *does* straddle
consecutive tars (verified: `vR1qObzkc3M`, `vRD7S-nfb1U`, `vRjqre-RNf8` appear in both the
tail of `EN-B000000.tar` and the head of `EN-B000001.tar`).

### Sidecar fields

```json
{"id": "EN_B00000_S00040_W000003", "wav": "...", "text": " Like I said, ...",
 "duration": 7.932, "speaker": "EN_B00000_S00040", "language": "en", "dnsmos": 3.2252}
```

Seven fields, nothing else. **The `start`/`end` offsets Emilia-Pipe computes internally were
dropped from the release**, so segment adjacency cannot be reconstructed, and there is no
diarisation confidence or speaker count. Emilia-YODAS keeps the raw pyannote label in the
speaker field (`EN_tKvmUvxYZXI_SPEAKER_00`) plus a `phone_count`.

### A speaker ID is a per-source pseudo-label

Diarisation runs per audio file (`preprocessors/Emilia/main.py`, `pyannote/speaker-diarization-3.1`),
with **no speaker verification or clustering across files**. So a speaker ID is one voice in
one recording:

- Labels are reliable as *same voice* groupings (all clips of an ID are one person).
- They are **not** reliable as *different voice* separations — one person appearing in two
  sources gets two unrelated IDs, in different batches entirely.
- Therefore a naive train/test split by speaker ID **leaks the same voice across both sides**.
  See [§5](#5-linking-the-same-person-across-sources).

### `en-b000000` at a glance

| | |
|---|---|
| clips / sidecars | 24,932 each (1.9 GB) |
| total | 69.4 h, mean 10.02 s |
| speaker IDs | 1,000 (median 8 clips each, max 332) |
| clips ≥ 6 s | 18,391 |
| `dnsmos` | min 3.00, p1 3.01, median 3.26 — Emilia pre-filtered at 3.0, so **it has no dynamic range left and cannot discriminate anything** |

`ls data/Emilia/en-b000000/*.mp3` fails with "argument list too long"; use `find` or glob in Python.

---

## 2. Diarisation leakage — the dominant defect

**Roughly a third of the batch has a second audible speaker.** Measured by ear on 20
score-blind random clips ≥6 s: 6 positives, i.e. **30%** (95% CI ≈ 15–52%). An independent
model-side estimate agrees: 36.5% of 200 random clips flag.

Nothing else in the cascade sees it. All six original stages score channel properties, and a
two-person exchange on one microphone passes every one of them. Measured on 600 random clips with the
windowed-embedding score of [§3](#3-what-does-not-detect-a-second-speaker):
`corr(score, dnsmos) = -0.13`, `corr(score, duration) = -0.04`.

`tools/metrics/multi_speaker.py` uses `pyannote/segmentation-3.0` (gated — accept the licence
on the model page) at **0.045 s/clip** (200 clips in 9 s), ~20 min for a whole shard. Two scores off one forward pass:

- **`second`** — seconds the second-most-active speaker talks, overlapping or not. Recall-first.
- **`overlap`** — seconds two speakers talk at once. A strict subset; the confident signal.

### Calibration

| set | `second` AUC | `overlap` AUC |
|---|---|---|
| 20 blind random clips (6 pos) | 0.917 | 0.940 |
| 29 blind clips stratified over the score range (17 pos / 10 neg decisive) | 0.847 | 0.882 |

On the stratified pool **every clip with `overlap > 0` was a true positive — 13/13, precision
1.00** (recall 0.76), and `second > 0.1` caught all 17 (precision 0.89). Across both blind
pools, **all 23 confirmed positives scored `second ≥ 0.14`**; no negative exceeded 0.08.

The score is bimodal, so the exact threshold barely matters (200 random clips ≥6 s):

| gate | flag rate |
|---|---|
| `second > 0` | 38.5% |
| `second > 0.1` | 37.0% |
| `second > 0.2` | 36.5% |
| `overlap > 0` | 27.5% |

Ambiguity lives entirely in `second > 0, overlap = 0` — clean turn-taking versus one
expressive speaker. Both clips the human labeller was unsure about had `overlap = 0.00`.

Shipped default is the recall-first gate: `multi_speaker_max: {second: 0.0, overlap: 0.0}`.
Raise `second` to ~0.5 and keep `overlap` at 0.0 for a precision-first gate.

### Flag rate by duration — the metric is uncalibrated below 6 s

Every labelled clip was ≥ 6 s, because both scans sampled `duration >= 6.0`. Flag rates fall
off sharply below that, and it is not known how much is genuine versus model degradation on
short context (150 random clips per band):

| 3–4 s | 4–5 s | 5–6 s | 6–10 s | 10–20 s |
|---|---|---|---|---|
| 16.7% | 26.7% | 30.7% | 35.3% | 41.3% |

`quality_filtering_emilia.yaml` sets `min_duration: 3.0`, i.e. it admits clips into a range
where this stage has never been validated.

### GOTCHA: the resampler changes the scores by ~30%

Emilia is 24 kHz, the model wants 16 kHz. `torchaudio.functional.resample` with its default
kernel inflated both scores by about 30% (4.58 → 6.11 s on one clip) — the aliasing it leaves
near Nyquist reads as speaker activity. `multi_speaker.py` pins kaiser-best parameters, which
reproduce librosa/soxr values exactly. The CTC metric still uses the default kernel.

---

## 3. What does *not* detect a second speaker

Recorded so these are not re-attempted.

| approach | result |
|---|---|
| Metadata (`dnsmos`, `duration`, text) | Nothing. 3 of 24,932 transcripts contain quotes, 0 contain turn dashes — the ASR transcribes both speakers inline |
| CTC `gap_speech` / `trail_speech` | Only catches an intruder the ASR *dropped* (a backchannel). Emilia's ASR ran on the cut segment, so the second voice is usually in the transcript and aligns fine |
| WavLM x-vector windows + speaker anchor | **AUC 0.655 on unbiased data** (0.828 on a score-selected set — the gap is pure selection bias). At its "recall 1.00" point it achieved recall 0.17 on random clips |
| Shorter windows for the above | Monotonically worse: 0.75 s → 0.715, 1.0 → 0.730, 1.5 → 0.749, 2.0 → 0.801. Embedding quality degrades faster than short intrusions are recovered |
| Fitting a classifier on the two scores | LOO-CV 0.766, worse than the unfitted product 0.818. Do not fit on ~40 points |
| F0 shift across the clip as corroboration | Unreliable in both directions — disagreed with the human on 4 of 12 clips. It tracks intonation, not identity |

---

## 4. The blind spot, and the source-level fix

`pyannote/segmentation-3.0` systematically merges **clean, non-overlapping turn-taking between
acoustically similar voices** into a single speaker. Two confirmed examples:

```
EN_B00000_S08620_W000000   4.35 s   second=0.00   "...Do you watch yourself? I can't watch myself."
EN_B00000_S07200_W000014  11.42 s   second=0.00   "...What'd you eat? Biscuits and lollies."
```

Not a threshold or framing artifact: mean powerset mass 0.575 on silence and 0.409 on *one*
speaker, **no frame of 589 picks a two-speaker class, max two-speaker posterior 0.023**, and
raw / zero-pad / reflect-pad / tiled / peak-normalised all return exactly 0.00. The WavLM
method misses them too (product 0.015 against a 0.05 gate) — both models judge the voices
acoustically close. In one conversational-biased pool, **4 of 23 gate-passed clips (~17%) still
contained a second speaker**.

### The fix: judge the source, not the clip

An Emilia speaker is one voice in one recording, so the leak is a property of the *source*.
Over 19 labelled clips with leave-one-out speaker rates:

| predictor | AUC |
|---|---|
| the clip's own `second` score | 0.622 |
| **its speaker's flag rate** | **0.900** |

Rule `flag if own > 0 OR speaker rate ≥ 0.5`: recall 1.00, and **0 of 9 labelled negatives
dropped**. Both leaks above are caught (`S07200` rate 0.79 over 19 clips, `S08620` 1.00 over 2).

Independent confirmation from a later, unrelated experiment: 5 clips a listener spotted as
multi-speaker after passing the per-clip gate belonged to speakers with rates **0.58, 0.67,
0.75, 0.75, 0.91** — all dropped by this rule.

Cost, on 60 random speakers (991 clips): per-clip flagging removes 31.7%; adding the source
rule at 0.5 takes it to **39.9%** (+8.2 pp). Thresholds 0.3 → 50.4%, 0.6 → 37.5%, 0.8 → 33.2%.

Implemented as a second pass in `scripts/filter/emilia.py`, config keys
`source_rejection_enabled` / `source_max_flag_rate` (0.5) / `source_min_clips` (3).

**GOTCHA:** the cascade stops at its first rejection, so a clip rejected by NISQA never gets a
multi-speaker score. The rate must be taken over clips that *reached* the stage — otherwise a
speaker whose clips mostly died on quality looks clean for never having been scored
(`reached()` in that script derives this from the pipeline's real stage order).

---

## 5. Linking the same person across sources

Use **`pyannote/wespeaker-voxceleb-resnet34-LM`** — ungated, loads via `PretrainedSpeakerEmbedding`
from the already-installed `pyannote.audio`, 256-d, **~0.017 s/clip**
(634 speakers × 4 clips, decode included, in 42 s). Embed whole clips (4–20 s, truncated to 8 s), L2-normalise, average per speaker, cosine.

**Do not use `microsoft/wavlm-base-plus-sv` for this.** Its between-speaker median is 0.71 with
p99 0.984 — it links 3% of *all* pairs at 0.95+. Mean-centring the embeddings helps but not
enough.

Distributions over 634 speakers with ≥4 clips (200,661 pairs):

```
within-speaker  (same source, split halves):  median 0.869   p5 0.728   p1 0.574
between-speaker (all cross-source pairs):     median 0.092   p99 0.835  p99.9 0.959
```

### Two thresholds

| similarity | meaning |
|---|---|
| **≥ 0.95** | duplicated audio, not a speaker link — see [§6](#6-duplicated-sources) |
| **≥ 0.86** | same person, different source |
| < 0.86 | different people |

The 0.86 cut comes from 12 blind listening pairs with disjoint transcripts: **AUC 1.000**, with
positives at 0.865–0.928 and negatives at 0.791–0.854. Narrow evidence (9 decisive labels) but
perfectly separated, and it puts the originally-noticed pair `S08710 ↔ S08350` (0.891) inside
the same-speaker band.

### Clustering gotchas

- **Use complete linkage.** Average linkage chained 634 speakers into a 57-ID cluster via weak
  bridges; complete linkage caps it at 22 (and that 22-ID cluster was real duplication, with a
  *minimum* pairwise similarity of 0.95).
- **Build centroids only from source-rule survivors.** A centroid computed from a clip holding
  two voices sits between them and bridges unrelated speakers.
- **Remove duplicates before linking**, or they produce large, perfectly cohesive clusters that
  look like prolific speakers.

---

## 6. Duplicated sources

The same recording appears under many speaker IDs. In `en-b000000`:

- **50 transcripts appear under more than one speaker ID**, covering **270 clips (1.1% of the
  batch)** and **65 of 1,000 speaker IDs**.
- The largest case is one podcast ("Food Junkies") whose fixed intro/outro was captured in
  **22 separate episodes**, each given its own speaker ID. Five of the IDs involved are
  **100% boilerplate** — every clip they contain is duplicated elsewhere.

It is the same audio, re-encoded — not the same script read twice:

```
S06150_W000000 vs S07570_W000000   18.044 s (identical)   aligned peak corr 0.9967 (lag -17 ms)
S06150_W000001 vs S07850_W000001    6.364 s               aligned peak corr 0.9979
S06400_W000000 vs S06450_W000000   18.044 s               aligned peak corr 0.9990
```

**md5 never matches** and a naive un-aligned sample correlation is ~0, so checksums and raw
correlation both miss it. Detection that works:

- normalised exact transcript match — caught all of it here, essentially free;
- **wespeaker similarity ≥ 0.95 — 230 of 230 such pairs shared a transcript (100%)**, so it
  doubles as a transcript-free duplicate detector;
- aligned cross-correlation to confirm.

No embedding model can separate "same person, different recording" from "same recording twice";
that is not a speaker-verification question. Deduplicate first.

---

## 7. Practical notes

- `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` are gated (403 until the
  licence is accepted); `pyannote/wespeaker-voxceleb-resnet34-LM` is **not**.
- `soundfile` reads Emilia's mp3s directly at 24 kHz — no ffmpeg step needed.
- No transcript in this batch contains `|` or a backslash, so the pipe-delimited CSV never
  needs escaping.
- Per-dataset configs: `configurations/quality_filtering_emilia.yaml` vs `_libritts.yaml`. A
  missing `--config` falls back to built-in defaults **silently**.
- Emilia speakers carry a median of 8 clips, so LibriTTS's `--min-per-speaker 7` keeps far
  fewer speakers here than on audiobooks; re-derive per shard.
