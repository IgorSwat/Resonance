"""Prosodic diversity selection: stratify -> water-fill -> facility-location within cells.

Scans LibriTTS, scores every clip with the prosody pipeline, then selects a subset whose
prosody-cell histogram is as flat as the data allows, under a per-speaker cap. Compares the
result against a random subset of the same size.
"""

import collections, pathlib, random, shutil, time
import numpy as np, soundfile as sf
from prosody import f0_contour, reference_pitch, prosody_features

N_SPEAKERS, N_CLIPS_PER_SPEAKER = 500, 12
MIN_DURATION = 2.0
OCTAVE_GUARD = (0.67, 1.5)
BUDGET = 80                      # clips to select
SPEAKER_CAP = 2                  # at most this many clips per speaker
EXPR_EDGES = [0, 10, 35, 65, 90, 100]   # percentiles: tails get their own cells
RATE_BINS = 4                           # x 3 final-slope classes = 60 cells
SLOPE_LEVEL = 5.0                # |st/s| below this counts as level
OUT = pathlib.Path("tmp")

FEATURES = ["st_range", "st_slope_std", "st_dct1", "st_dct2", "st_dct3", "st_dct4",
            "st_final_slope", "voiced_frac", "rms_db_std", "voiced_onsets_per_s", "n_phrases"]


def voiced_onset_rate(f0, duration):
    """Voiced-group onsets per second — a speaking-rate proxy, standing in for the CTC aligner."""
    voiced = (~np.isnan(f0)).astype(np.int8)
    return int((np.diff(np.r_[0, voiced]) == 1).sum()) / duration


# ---------------------------------------------------------------- scan
random.seed(21)
by_speaker = collections.defaultdict(list)
for p in sorted(pathlib.Path("data/librispeech/audio").iterdir()):
    by_speaker[p.name.split("_")[0]].append(p)
speakers = sorted(random.sample(
    [s for s, v in by_speaker.items() if len(v) >= N_CLIPS_PER_SPEAKER], N_SPEAKERS))

t0 = time.time()
pass1 = []
for s in speakers:
    for p in random.sample(by_speaker[s], N_CLIPS_PER_SPEAKER):
        info = sf.info(p)
        if info.duration < MIN_DURATION:
            continue
        y, sr = sf.read(p, dtype="float32")
        audio = {"audio": y.astype(np.float64), "sample_rate": sr}
        pass1.append((s, p, audio, reference_pitch(f0_contour(audio))))
print(f"pass 1: {len(pass1)} clips from {len(speakers)} speakers, {time.time()-t0:.0f}s")

refs = {}
for s in speakers:
    medians = [c[3]["median_hz"] for c in pass1 if c[0] == s and c[3]]
    if medians:
        refs[s] = float(np.median(medians))

t0 = time.time()
rows, rejected = [], collections.Counter()
for s, path, audio, r in pass1:
    if r is None or s not in refs:
        rejected["no reference"] += 1; continue
    ref = refs[s]
    if not OCTAVE_GUARD[0] * ref <= r["median_hz"] <= OCTAVE_GUARD[1] * ref:
        rejected["octave guard"] += 1; continue
    f0 = f0_contour(audio, floor=max(50.0, 0.55 * ref), ceiling=min(500.0, 1.9 * ref))
    row = prosody_features(f0, audio, ref)
    if row is None:
        rejected["no features"] += 1; continue
    duration = len(audio["audio"]) / audio["sample_rate"]
    row |= {"speaker": s, "path": path, "ref_hz": ref, "dur": duration,
            "voiced_onsets_per_s": voiced_onset_rate(f0, duration)}
    rows.append(row)
print(f"pass 2: {len(rows)} scored clips, {dict(rejected)}, {time.time()-t0:.0f}s")

# per-speaker z-score of the energy block (eta^2 0.70 -> it is a recording property)
for s in speakers:
    sub = [r for r in rows if r["speaker"] == s]
    if len(sub) < 3:
        continue
    v = np.array([r["rms_db_std"] for r in sub])
    for r, z in zip(sub, (v - v.mean()) / (v.std() + 1e-9)):
        r["rms_db_std"] = float(z)

# ---------------------------------------------------------------- cells
def quantile_bin(values, n):
    edges = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, edges)

expr_values = np.array([r["st_range"] for r in rows])
expr = np.digitize(expr_values, np.percentile(expr_values, EXPR_EDGES[1:-1]))
rate = quantile_bin(np.array([r["voiced_onsets_per_s"] for r in rows]), RATE_BINS)
slope = np.array([0 if r["st_final_slope"] < -SLOPE_LEVEL else
                  2 if r["st_final_slope"] > SLOPE_LEVEL else 1 for r in rows])
for r, e, a, sl in zip(rows, expr, rate, slope):
    r["cell"] = (int(e), int(a), int(sl))

cells = sorted({r["cell"] for r in rows})
members = {c: [r for r in rows if r["cell"] == c] for c in cells}
print(f"\n{len(cells)}/{(len(EXPR_EDGES)-1)*RATE_BINS*3} cells occupied | "
      f"sizes: min {min(len(v) for v in members.values())}, "
      f"median {int(np.median([len(v) for v in members.values()]))}, "
      f"max {max(len(v) for v in members.values())}")

# ---------------------------------------------------------------- water-fill
def water_fill(counts, budget):
    take = np.zeros(len(counts), dtype=int)
    remaining, active = budget, counts > 0
    while remaining > 0 and active.any():
        quota = max(1, remaining // active.sum())
        give = np.minimum(quota, counts - take) * active
        if not give.sum():
            break
        order = np.argsort(-give)          # spend the tail of the budget deterministically
        for i in order:
            if remaining <= 0:
                break
            g = min(give[i], remaining)
            take[i] += g; remaining -= g
        active &= take < counts
    return take

# capacity is limited by the speaker cap inside each cell, not just by raw membership
capacity = np.array([
    sum(min(SPEAKER_CAP, n) for n in collections.Counter(
        r["speaker"] for r in members[c]).values()) for c in cells])
quota = water_fill(capacity, BUDGET)
print(f"water-fill: {quota.sum()} clips over {(quota>0).sum()} cells | "
      f"quota min {quota[quota>0].min()}, max {quota.max()} | "
      f"cells capped by availability: {(quota==capacity).sum()}")

# ---------------------------------------------------------------- within-cell selection
M = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
M = (M - M.mean(0)) / (M.std(0) + 1e-9)
for r, v in zip(rows, M):
    r["vec"] = v

def facility_location(candidates, k, cap_counter):
    """Greedy: each pick maximizes the gain in sum-of-similarity-to-nearest-selected."""
    picked, best = [], None
    X = np.array([c["vec"] for c in candidates])
    D = -np.linalg.norm(X[:, None] - X[None], axis=-1)     # similarity = negative distance
    available = [i for i in range(len(candidates))]
    while len(picked) < k and available:
        if best is None:
            gains = [D[i].sum() for i in available]
        else:
            gains = [np.maximum(best, D[i]).sum() for i in available]
        i = available[int(np.argmax(gains))]
        chosen = candidates[i]
        picked.append(chosen)
        cap_counter[chosen["speaker"]] += 1
        best = D[i] if best is None else np.maximum(best, D[i])
        available = [j for j in available if j != i
                     and cap_counter[candidates[j]["speaker"]] < SPEAKER_CAP]
    return picked

cap_counter = collections.Counter()
selected = []
for c, k in sorted(zip(cells, quota), key=lambda ck: -ck[1]):
    if k:
        selected += facility_location(members[c], int(k), cap_counter)
print(f"selected {len(selected)} clips\n")

# ---------------------------------------------------------------- evaluation
random.seed(99)
baseline = random.sample(rows, len(selected))

def entropy(subset):
    counts = np.array(list(collections.Counter(r["cell"] for r in subset).values()), float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def report(name, subset):
    rng = np.array([r["st_range"] for r in subset])
    onset = np.array([r["voiced_onsets_per_s"] for r in subset])
    dur = np.array([r["dur"] for r in subset])
    n_cells = len({r["cell"] for r in subset})
    n_spk = len({r["speaker"] for r in subset})
    rising = np.mean([r["st_final_slope"] > SLOPE_LEVEL for r in subset])
    print(f"  {name:10} cells {n_cells:2d}/60  entropy {entropy(subset):.2f}  |  "
          f"st_range {rng.mean():5.2f} +/- {rng.std():4.2f} [{rng.min():5.2f}, {rng.max():5.2f}]  |  "
          f"rate sd {onset.std():4.2f}  |  spk {n_spk:3d}  |  rising {rising:.2f}  |  "
          f"dur {dur.mean():4.1f}s +/- {dur.std():4.1f}")

print(f"max possible cell entropy = {np.log2(60):.2f} bits")
report("full pool", rows)
report("random", baseline)
report("diversity", selected)

print("\nst_range decile occupancy (how flat is the expressiveness marginal?)")
edges = np.percentile([r["st_range"] for r in rows], np.linspace(0, 100, 11))
for name, subset in (("full pool", rows), ("random", baseline), ("diversity", selected)):
    h = np.histogram([r["st_range"] for r in subset], bins=edges)[0]
    print(f"  {name:10} " + " ".join(f"{x:4d}" for x in h) +
          f"   (ideal flat = {len(subset)//10})")

# ---------------------------------------------------------------- write
for f in OUT.iterdir():
    if f.is_file():
        f.unlink()
for r in sorted(selected, key=lambda r: (r["cell"], -r["st_range"])):
    e, a, sl = r["cell"]
    name = (f"libritts_div_e{e}r{a}s{sl}_range{r['st_range']:05.2f}"
            f"_rate{r['voiced_onsets_per_s']:04.1f}_fslope{r['st_final_slope']:+06.1f}"
            f"_ref{r['ref_hz']:03.0f}hz_{r['dur']:04.1f}s_{r['path'].name}")
    shutil.copy2(r["path"], OUT / name)
print(f"\nwrote {len(selected)} clips to {OUT}/  ({sum(r['dur'] for r in selected)/60:.1f} min)")
