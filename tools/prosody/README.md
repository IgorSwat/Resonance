# Prosody-based dataset selection — how it works

Explains the code in this directory: how a `.wav` becomes a row of numbers describing *how it
was spoken*, and how those rows are used to pick a training subset with wide prosodic variety
instead of the usual pile of neutral narration. See `knowledge/data_filtering.md` §2 for where
this fits in the wider filtering cascade.

Everything here is exploratory — nothing is wired into `tools/` or `scripts/` yet.

```
prosody.py          feature extraction (the core)
test_prosody.py     validation: synthetic ground truth, 8 speakers, LJSpeech questions
test_scale.py       60-speaker run: variance decomposition, correlations, defect hunting
pick_expressive.py  writes the least/most expressive clips to ./tmp for listening
select_diverse.py   the selection algorithm, with a random baseline to beat
```

Requires `pip install praat-parselmouth`. Run from the repo root:

```
PYTHONPATH=scratch python scratch/select_diverse.py
```

`pick_expressive.py` and `select_diverse.py` both **delete everything in `./tmp`** before writing.

---

## The problem

Take 10,000 hours of audiobooks and pick 2,000 at random. The subset has the same prosody
distribution as the source: mostly flat declarative narration, a narrow band of speaking rates,
almost no questions. A model trained on it can only produce that, because nothing else varied
enough to become a learnable axis.

The fix is to measure how each clip is spoken, then deliberately pick a subset that spans a wide
range instead of piling onto the mode. That needs three things: a measurement, a way to make
measurements comparable across speakers, and a selection rule.

---

## Step 1 — Extract the pitch contour

Prosody is what's left of an utterance once you remove which words were said. It lives in three
physical signals: **pitch** (F0), **energy**, and **timing**. Pitch carries the most and is the
hardest to handle, so it dominates the code.

`f0_contour()` runs Praat's autocorrelation pitch tracker over the waveform and returns one
number per 10 ms frame:

```python
f0 = f0_contour({"audio": y, "sample_rate": sr})
# array([nan, nan, 118.3, 119.1, 121.4, ..., nan, nan])
```

Frames are `NaN` where the speech is unvoiced — during `/s/`, `/f/`, `/t/`, and silence. Typically
40–70% of frames come back voiced. **Every numpy reduction on this array returns NaN unless you
mask first**, which is the first thing the code does.

Why Praat rather than the obvious `librosa.pyin`? Measured on this repo's own data, 63 s of audio
on one core:

| tracker | speed | notes |
|---|---|---|
| **parselmouth (Praat)** | **2076× realtime** | what we use |
| `pyworld.dio` | 352× | built for vocoding |
| `librosa.pyin` | 13.6× | accurate, 150× slower |
| `pyworld.harvest` | 15.7× | slow *and* wrong here (see below) |
| `torchcrepe` tiny, CPU | 4.1× | needs a GPU to be practical |

Praat agrees with pyin to within 0.4 semitones on the statistics we actually use, so the speedup
is free. Harvest was rejected on quality, not speed: it reported pitch ranges of 18–22 semitones
where the other two said 6–9, because it force-fits a continuous contour for a vocoder instead of
deciding what is voiced.

---

## Step 2 — Normalize away the speaker

This is the step everything else depends on.

Suppose you describe a clip as "mean pitch 210 Hz, std 28 Hz". You have mostly described the
speaker's larynx. Cluster on those numbers and you get two clusters — higher voices and lower
voices — and you have rediscovered speaker identity, which the x-vector already gives you for free.

Two conversions fix it.

**Log scale.** Pitch is heard multiplicatively: 100→200 Hz and 200→400 Hz are both one octave and
sound like equal-sized moves. In Hz, a high voice automatically looks more variable than a low one,
purely as an artifact.

**Speaker-relative reference.** Divide by the speaker's own habitual pitch before taking the log:

```
st(t) = 12 · log2( f0(t) / speaker_median_hz )
```

Worked example, two clips that are prosodically identical but spoken by different people:

| | raw F0 | in Hz | as semitones vs own median |
|---|---|---|---|
| bass, median 100 Hz | 89 → 119 Hz | std ≈ 11 Hz | −2.0 → +3.0 st |
| soprano, median 200 Hz | 178 → 238 Hz | std ≈ 22 Hz | −2.0 → +3.0 st |

In Hz the soprano looks twice as expressive. In speaker-relative semitones they are correctly
identical. Now the number means "how much did this person move *for them*", which is behaviour,
not anatomy.

**Use the speaker's pooled median across all their clips, never the clip's own median.** Per-clip
normalization would set every clip's mean to zero and erase the difference between that speaker's
animated and flat recordings — which is exactly the signal we're after. This is why the pipeline
needs two passes.

---

## Step 3 — Turn a contour into a fixed-width row

A 3-second clip has ~150 voiced frames, a 12-second clip ~600. To compare, cluster, and sample
you need the same number of columns for every clip. `prosody_features()` produces them in
three groups.

### Group A — how much pitch moves

```python
"st_std":        observed.std()
"st_range":      P95 − P5            # the primary expressiveness axis
"st_slope_std":  diff(observed).std()  # jumpy vs smooth
"voiced_frac":   fraction of frames voiced
```

`st_range` uses percentiles rather than min/max so one bad frame can't dominate. On real LibriTTS
it runs about 2.5 (near-monotone) to 19 semitones (animated), mean ≈ 10.

### Group B — the *shape* of the contour

Two clips can have the same range while one rises and the other falls. To capture shape
independent of duration, the contour is resampled to a fixed 64 points and run through a DCT:

```python
grid = np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, len(st)), st)
coeffs = dct(grid, type=2, norm="ortho")[1:5]     # st_dct1 .. st_dct4
```

Each coefficient is the weight of a basis shape:

```
c0  overall level        (dropped — that's just mean pitch, i.e. speaker identity)
c1  rise vs fall         negative = rises across the utterance
c2  hump vs valley
c3, c4  finer wiggle
```

The resample is what makes a 3 s and a 12 s clip comparable. Dropping `c0` is what keeps speaker
pitch out of a shape descriptor. Both matter; skipping either produces a feature that silently
measures something else.

`st_final_slope` is computed separately: a straight-line fit over the last 300 ms of the final
phrase, in semitones per second. This is the most linguistically loaded number in the row —
phrase-final rise vs fall is the question/statement distinction.

**Phrases, not utterances.** Unvoiced gaps shorter than 200 ms are just consonants and get
interpolated across; longer gaps are phrase boundaries and *split* the contour. Interpolating a
straight line across a 700 ms pause fabricates a contour nobody spoke and pollutes every DCT
coefficient.

### Group C — energy

`rms_db_std` and `rms_db_range` over the speech frames. See the warning in the "what went wrong"
section below — these turned out to measure the recording chain, not the speaking.

---

## Step 4 — The two-pass structure

Because the reference pitch needs all of a speaker's clips, you cannot do this in one sweep:

```
PASS 1  for every clip:  extract F0 -> store its median Hz and voiced-frame count
        per speaker:     reference = median of those clip medians

PASS 2  for every clip:  re-extract F0 with a speaker-adapted range
                         convert to semitones vs the reference
                         compute the feature row
```

Both passes together run at ~2500× realtime, so 10,000 hours of F0 costs about 4 core-hours.
Audio *decoding* is now the bottleneck, not pitch tracking.

---

## Does it actually work? Three tests

`test_prosody.py` runs all three.

### Test A — synthetic contours with a known answer

Generate a voiced signal whose pitch follows a contour we control, then check what comes back:

```
flat 120 Hz                st_std 0.000   st_range 0.001
vibrato depth 0.5 st       st_std 0.347   (predicted 0.354)
vibrato depth 1.0 st       st_std 0.695   (predicted 0.707)
vibrato depth 2.0 st       st_std 1.400   (predicted 1.414)
vibrato depth 4.0 st       st_std 2.819   (predicted 2.828)
ramp +6 st over 3 s        st_final_slope +2.00 st/s    dct1 −13.74
ramp −6 st over 3 s        st_final_slope −2.00 st/s    dct1 +13.74
```

Everything within ~1% of the analytic prediction, and the shape features are correctly signed.
The measurement layer does what it claims.

### Test B — 60 real speakers: is it measuring the speaker or the speaking?

The key diagnostic is **η² — the share of a feature's variance explained by speaker identity**.
Low is good for a prosody feature; high means you re-derived the x-vector.

```
median_hz        0.826    <- speaker identity, as intended
hz_std           0.552    <- raw pitch spread, before normalization
st_std           0.431    <- after semitone + speaker-reference normalization
st_range         0.400
st_dct1          0.134    <- contour shape: essentially speaker-free
st_final_slope   0.128
rms_db_std       0.696    <- problem, see below
```

Normalization removes a fifth of the speaker share, and the shape features land near 0.1. The
residual in `st_range` is legitimate — some readers genuinely *are* more animated. The check that
settles it:

```
mean within-speaker sd of st_range     3.33
between-speaker sd of speaker means    2.91
```

The feature varies more *inside* a speaker than across speakers, so it carries per-clip
information the speaker embedding does not.

### Test C — does it recover a linguistic fact?

40 LJSpeech question-final utterances against 40 matched declaratives, one speaker so nothing
else varies:

```
st_final_slope   question  +2.51    statement −23.67    Cohen d +0.66
st_dct1          question  +0.58    statement  +8.71    Cohen d −0.79
st_std           question   3.99    statement   4.17    Cohen d −0.18
st_range         question  12.18    statement  12.24    Cohen d −0.02
fraction with a rising final:  questions 0.68   statements 0.23
```

Questions rise, statements fall, at d ≈ 0.7–0.8, with no supervision anywhere in the pipeline.

Just as informative: `st_std` and `st_range` show **no** separation. They measure expressiveness;
`dct1` and `final_slope` measure contour shape. The groups are genuinely independent, which is what
you want from a diversity feature space.

---

## What went wrong, and the fixes

Three defects that only appeared at 60-speaker scale. All three are in
`pick_expressive.py` / `select_diverse.py`; `prosody.py` is still the unfixed version the tests
were written against.

### 1. Octave errors corrupt the speaker reference

A fixed 60–400 Hz search range lets a ~95 Hz male voice be tracked at 190 or 354 Hz. Speaker 6694's
clip medians:

```
79 82 83 86 87 88 88 89 90 91 93 93 94 96 99 104 108 121 125 129 | 168 176 190 206 354
                                                                   ^ doubled
```

**6 of 60 speakers** showed a within-speaker spread above 1.8×. Since the reference divides every
semitone value, a bad reference corrupts that speaker's entire feature row.

Fix — a robust reference plus an adapted search range:

```python
ref = np.median(clip_medians)                    # median, not mean: doubled clips can't drag it up
f0  = f0_contour(audio, floor=max(50, 0.55*ref), ceiling=min(500, 1.9*ref))
```

```
fixed 60-400 Hz     median 1.37x  p90 1.80x  max 4.45x  | >1.8x: 6 speakers
adapted 0.55-1.9x   median 1.35x  p90 1.59x  max 2.21x  | >1.8x: 3 speakers
```

Plus a per-clip guard rejecting anything outside 0.67–1.5× the reference. On a 500-speaker scan
this rejected 27 of 5254 clips.

### 2. Short clips fake expressiveness

The *mean* of `st_range` is unbiased with duration (r = +0.077) but the *variance* is not:

```
duration    n     mean   sd     p95
0-1  s      18   11.77  7.77  26.69
2-4  s     194   11.16  4.69  20.53
8+   s     297   12.21  4.21  20.39
```

Clips under 2 s are 12% of the data but **20% of the top-5% by `st_range`**. Since selection
targets that tail, short clips get picked on measurement noise. Fix: require ≥ 2 s.

The combined effect of fixes 1 and 2 is large. Before them, the "most expressive" clips reached
`st_range` 29.4; after, the true maximum is 19.4. **The entire 20–29 semitone tail was tracker
error, not expressive speech** — and it would have been selected preferentially.

### 3. The energy block measures the microphone

`rms_db_std` has η² = 0.70 — it is mostly determined by who recorded the clip. Across 60 speaker
means: mean 10.05, sd 1.34, but speaker 159 sits at **2.03**, ~6 sd low. That is a compressed
recording chain, not a speaking style. Fix: z-score the energy block per speaker, or drop it.

### Redundant columns

```
st_std          st_range              r = +0.91   <- keep st_range only
rms_db_std      rms_db_range          r = +0.82   <- keep one
st_std          st_slope_std          r = +0.67
```

---

## Step 5 — Selecting a diverse subset

`select_diverse.py`. The useful framing: **you are not maximizing distance, you are flattening a
distribution under constraints.** Maximizing distance picks outliers, and outliers in in-the-wild
data are mostly defects.

### 5a. Stratify on axes you can name

Three interpretable axes crossed into cells:

```
expressiveness  (st_range)              5 bins
speaking rate   (voiced onsets/sec)     4 bins
final slope     falling / level / rising 3 classes
                                        = 60 cells
```

Named axes rather than a k-means over the raw vectors, because later you need to be able to say
*which* bins filled up. "Cluster 17 is underrepresented" is not something you can act on.

### 5b. Water-fill the budget

Cells are wildly unequal — the widest cells held 9 clips, the largest 264. An equal quota of
`budget/60` would exhaust the rare cells and leave the budget unspent. The maximum-entropy
allocation is water-filling: give every cell an equal quota, let cells that can't fill it
contribute what they have, and redistribute the surplus.

```python
def water_fill(counts, budget):
    take, remaining, active = np.zeros(len(counts), int), budget, counts > 0
    while remaining > 0 and active.any():
        quota = max(1, remaining // active.sum())
        give = np.minimum(quota, counts - take) * active
        ...
```

Capacity per cell is computed **after** the per-speaker cap, not before — otherwise the allocation
promises clips the cap won't let you take.

### 5c. Pick within each cell by coverage

Greedy facility-location: each pick maximizes the gain in sum-of-similarity-to-nearest-selected.
It rewards *representativeness*, so it covers a cell rather than hugging its edges. The
per-speaker cap is enforced during the greedy loop, not afterwards — post-hoc filtering breaks
the allocation you just computed.

---

## The trap: never stratify on quantiles of the feature you want to flatten

The first version used **quartile** bins for expressiveness. Result:

```
random     cells 31/48  entropy 4.65  |  st_range 10.32 +/- 3.08  |  rising 0.12
diversity  cells 48/48  entropy 5.52  |  st_range 10.03 +/- 2.59  |  rising 0.33
```

Cell coverage improved a lot — but `st_range` **sd went down**, 3.08 → 2.59. The selection was
*less* varied than random on the axis that matters most.

The cause is structural. Facility-location picks the clip nearest each cell's centre, so quartile
bins guarantee four centroids and no tails. More fundamentally: a pool is uniform across its own
quantiles by construction, so stratifying on quantiles of a feature cannot flatten that feature —
there is nothing left to flatten.

Fix: bin at percentiles `[0, 10, 35, 65, 90, 100]` so the extreme deciles become their own cells
and water-fill is *obliged* to sample them.

```
                                                             
full pool  cells 60/60  entropy 5.48  |  st_range 10.04 +/- 2.81  |  spk 500  |  rising 0.17  |  dur 7.5s
random     cells 35/60  entropy 4.78  |  st_range 10.32 +/- 3.08  |  spk  74  |  rising 0.12  |  dur 7.5s
diversity  cells 60/60  entropy 5.82  |  st_range 10.50 +/- 3.25  |  spk  70  |  rising 0.34  |  dur 8.6s
```

Now it wins on every intended axis: full cell coverage against random's 35/60, entropy 5.82 of a
possible 5.91, `st_range` sd above random rather than below, and nearly 3× the rare rising-final
class. The expressiveness histogram becomes deliberately U-shaped — range preferred over
consistency, stated explicitly rather than hoped for:

```
              d1  d2  d3  d4  d5  d6  d7  d8  d9 d10
random         8   7   8  10   3   7  14   5   5  13
diversity     13   7   4   2   6   8   5   9  11  15
```

---

## Verify, or you have done nothing

A broken prosody pipeline fails silently — it still returns vectors, still clusters, still
selects, and the selection is just random. Always report, before and after:

- cell-histogram entropy (the headline number)
- the `st_range` marginal
- unique speakers per hour
- rising-final fraction

Then **listen**. `pick_expressive.py` writes the extremes to `./tmp`; that is what caught the
octave artifacts. If the "most expressive" clips are creak and tracker slips rather than animated
reading, no amount of entropy improvement means anything.

---

## Known limitations

- **Speaking rate is a proxy.** Voiced-group onsets per second stands in for phones/sec until the
  CTC aligner is wired in. It conflates rate with phonation. Same for phrase boundaries, which
  currently come from unvoiced gaps rather than real pauses.
- **No rhythm block.** Pause-length distribution and phone-duration variance need the aligner.
- **Selection drifts toward long clips** — 8.6 s mean against the pool's 7.5 s, sd 2.7 vs 4.2.
  Longer clips have more phrases and richer contours, so they win facility-location. Duration
  needs to be a fourth stratification axis, or the budget must be counted in seconds.
- **The very flattest clips are excluded** (`st_range` min 4.53 vs random's 2.84). Facility-location
  picks the centre even of the bottom-decile cell. Defensible, but it was the algorithm's choice.
- **Validated on English audiobooks only** — 108–225 Hz references, clean audio. Low male voices
  and noisy parliamentary audio are where octave errors concentrate. Re-run `test_scale.py` on the
  Polish corpora before trusting any threshold here.
- **Selection cannot manufacture range.** LibriTTS tops out at `st_range` 19.4. If the expressive
  tail isn't in the source, the water-fill occupancy report tells you how short you are — that is
  a sourcing problem, not a selection one.
