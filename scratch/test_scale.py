"""Test B at scale: 60 LibriTTS speakers x 15 clips."""

import random, time, collections, pathlib
import numpy as np, soundfile as sf
from prosody import f0_contour, reference_pitch, speaker_reference, prosody_features

N_SPEAKERS, N_CLIPS = 60, 15

paths = sorted(pathlib.Path("data/librispeech/audio").iterdir())
by_speaker = collections.defaultdict(list)
for p in paths:
    by_speaker[p.name.split("_")[0]].append(p)
random.seed(11)
speakers = sorted(random.sample([s for s, v in by_speaker.items() if len(v) >= N_CLIPS], N_SPEAKERS))

t0 = time.time(); decode_s = 0.0; audio_s = 0.0
cache = []
for s in speakers:
    for p in random.sample(by_speaker[s], N_CLIPS):
        t1 = time.time()
        y, sr = sf.read(p, dtype="float32")
        decode_s += time.time() - t1
        a = {"audio": y.astype(np.float64), "sample_rate": sr}
        audio_s += len(y) / sr
        f0 = f0_contour(a)
        cache.append((s, p.name, a, f0, reference_pitch(f0)))
total_s = time.time() - t0
print(f"{len(cache)} clips, {len(speakers)} speakers, {audio_s/60:.1f} min audio")
print(f"decode {decode_s:.1f}s | f0 {total_s-decode_s:.1f}s = {audio_s/(total_s-decode_s):.0f}x realtime\n")

refs = {}
for s in speakers:
    clips = [c[4] for c in cache if c[0] == s and c[4]]
    refs[s] = speaker_reference(clips) if clips else None

rows, skipped = [], collections.Counter()
for s, name, a, f0, r in cache:
    if r is None:
        skipped["too few voiced frames"] += 1; continue
    row = prosody_features(f0, a, refs[s])
    if row is None:
        skipped["no usable phrase / low voiced frac"] += 1; continue
    raw = f0[~np.isnan(f0)]
    row |= {"speaker": s, "path": name, "median_hz": float(np.median(raw)),
            "hz_std": float(raw.std()), "dur": len(a["audio"]) / a["sample_rate"]}
    rows.append(row)
print(f"rows: {len(rows)}/{len(cache)} ({100*len(rows)/len(cache):.1f}%)   skipped: {dict(skipped)}\n")

KEYS = ["st_std", "st_range", "st_p95", "st_slope_std", "st_dct1", "st_dct2", "st_dct3",
        "st_dct4", "st_final_slope", "st_phrase_slope_mean", "voiced_frac", "rms_db_std",
        "rms_db_range", "n_phrases"]

def eta2(key, rows):
    v = np.array([r[key] for r in rows], dtype=float)
    g = np.array([r["speaker"] for r in rows])
    grand = v.mean()
    between = sum(len(v[g == s]) * (v[g == s].mean() - grand) ** 2 for s in set(g))
    return between / ((v - grand) ** 2).sum()

print("feature distributions and speaker-variance share")
print(f"  {'feature':22} {'mean':>8} {'sd':>7} {'p5':>8} {'p95':>8}   eta^2")
for k in ["median_hz", "hz_std"] + KEYS:
    v = np.array([r[k] for r in rows], dtype=float)
    print(f"  {k:22} {v.mean():8.2f} {v.std():7.2f} {np.percentile(v,5):8.2f} "
          f"{np.percentile(v,95):8.2f}   {eta2(k, rows):.3f}")

print("\nreference-pitch stability (split-half over each speaker's clips, semitone difference)")
diffs = []
for s in speakers:
    clips = [c[4] for c in cache if c[0] == s and c[4]]
    if len(clips) < 8: continue
    a, b = speaker_reference(clips[::2]), speaker_reference(clips[1::2])
    diffs.append(abs(12 * np.log2(a / b)))
print(f"  n={len(diffs)}  median {np.median(diffs):.3f} st  p90 {np.percentile(diffs,90):.3f} st  max {max(diffs):.3f} st")

print("\nfeature correlation (|r| > 0.6 pairs — redundancy check)")
M = np.array([[r[k] for k in KEYS] for r in rows], dtype=float)
C = np.corrcoef(M.T)
pairs = [(abs(C[i, j]), KEYS[i], KEYS[j], C[i, j])
         for i in range(len(KEYS)) for j in range(i + 1, len(KEYS)) if abs(C[i, j]) > 0.6]
for _, a, b, r in sorted(pairs, reverse=True):
    print(f"  {a:22} {b:22} r={r:+.2f}")
if not pairs: print("  none")

print("\nspeakers with anomalous energy dynamics (rms_db_std)")
per = {s: np.mean([r["rms_db_std"] for r in rows if r["speaker"] == s]) for s in speakers}
v = np.array(list(per.values()))
print(f"  across speakers: mean {v.mean():.2f}  sd {v.std():.2f}  min {v.min():.2f}  max {v.max():.2f}")
for s, m in sorted(per.items(), key=lambda kv: kv[1])[:5]:
    print(f"  low  {s:>6} {m:6.2f}")

print("\nexpressiveness extremes (st_range)")
rows.sort(key=lambda r: r["st_range"])
for label, sel in (("flattest", rows[:5]), ("widest", rows[-5:])):
    print(f"  {label}:")
    for r in sel:
        print(f"    {r['path']:34} st_range {r['st_range']:6.2f}  st_std {r['st_std']:5.2f}  "
              f"voiced {r['voiced_frac']:.2f}  dur {r['dur']:5.1f}s")

print("\nwithin-speaker spread of st_range (does the feature vary inside a speaker?)")
w = [np.std([r["st_range"] for r in rows if r["speaker"] == s]) for s in speakers]
b = np.std([np.mean([r["st_range"] for r in rows if r["speaker"] == s]) for s in speakers])
print(f"  mean within-speaker sd {np.mean(w):.2f}   between-speaker sd of means {b:.2f}")
