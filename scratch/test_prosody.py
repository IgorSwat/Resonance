"""Validation of the prosody pipeline: synthetic ground truth, LibriTTS speakers, LJSpeech questions."""

import sys, time, random, collections
import numpy as np, soundfile as sf
from prosody import f0_contour, reference_pitch, speaker_reference, prosody_features

SR = 24000
rng = np.random.default_rng(0)


def synth(f0_hz, sr=SR):
    """Voiced signal whose F0 follows f0_hz exactly — the ground truth for test A."""
    phase = np.cumsum(2 * np.pi * f0_hz / sr)
    y = sum(np.sin(k * phase) / k for k in range(1, 21))
    return {"audio": (0.3 * y / np.abs(y).max() + 1e-3 * rng.standard_normal(len(y))), "sample_rate": sr}


def clip(y, sr=SR):
    return {"audio": np.asarray(y, dtype=np.float64), "sample_rate": sr}


def features(audio, ref=None):
    f0 = f0_contour(audio)
    r = reference_pitch(f0)
    if r is None:
        return None
    return prosody_features(f0, audio, ref or r["median_hz"]), r


print("=" * 78)
print("TEST A — synthetic contours with known ground truth")
print("=" * 78)
n = int(3.0 * SR)
t = np.arange(n) / SR

print("\nA1 flat 120 Hz (expect st_std ~ 0, st_range ~ 0)")
row, r = features(synth(np.full(n, 120.0)))
print(f"   median {r['median_hz']:6.1f} Hz   st_std {row['st_std']:.3f}   st_range {row['st_range']:.3f}")

print("\nA2 sinusoidal vibrato, 5 Hz, depth d semitones (expect st_std ~ d/sqrt2, st_range ~ 1.95d)")
print("   d      med_hz   st_std  (pred)   st_range  (pred)")
for d in (0.5, 1.0, 2.0, 4.0):
    f0 = 120 * 2 ** (d * np.sin(2 * np.pi * 5 * t) / 12)
    row, r = features(synth(f0))
    print(f"   {d:4.1f}  {r['median_hz']:7.1f}   {row['st_std']:6.3f}  ({d/np.sqrt(2):5.3f})   "
          f"{row['st_range']:7.3f}  ({1.95*d:5.3f})")

print("\nA3 linear ramps over 3 s (expect st_final_slope ~ +/-2 st/s, opposite dct1 signs)")
for name, span in (("rising  +6 st", 6.0), ("falling -6 st", -6.0)):
    f0 = 120 * 2 ** (np.linspace(0, span, n) / 12)
    row, r = features(synth(f0))
    print(f"   {name}:  st_final_slope {row['st_final_slope']:+6.2f} st/s   dct1 {row['st_dct1']:+7.2f}")

print("\nA4 flat contour, rising final 300 ms (the question case)")
for name, jump in (("final rise +4 st", 4.0), ("final fall -4 st", -4.0)):
    f0 = np.full(n, 120.0)
    tail = int(0.3 * SR)
    f0[-tail:] = 120 * 2 ** (np.linspace(0, jump, tail) / 12)
    row, r = features(synth(f0))
    print(f"   {name}:  st_final_slope {row['st_final_slope']:+6.2f} st/s")

print("\n" + "=" * 78)
print("TEST B — LibriTTS: 8 speakers x 12 clips, two-pass over real audio")
print("=" * 78)
import pathlib
paths = sorted(pathlib.Path("data/librispeech/audio").iterdir())
by_speaker = collections.defaultdict(list)
for p in paths:
    by_speaker[p.name.split("_")[0]].append(p)
random.seed(1)
speakers = random.sample([s for s, v in by_speaker.items() if len(v) >= 12], 8)

t0 = time.time()
cache, audio_s = {}, 0.0
for s in speakers:
    for p in random.sample(by_speaker[s], 12):
        y, sr = sf.read(p, dtype="float32")
        a = clip(y, sr)
        f0 = f0_contour(a)
        cache[p] = (s, a, f0, reference_pitch(f0))
        audio_s += len(y) / sr
extract_s = time.time() - t0

refs = {s: speaker_reference([c[3] for c in cache.values() if c[0] == s and c[3]]) for s in speakers}

rows, skipped = [], 0
for p, (s, a, f0, r) in cache.items():
    row = prosody_features(f0, a, refs[s]) if r else None
    if row is None:
        skipped += 1
        continue
    raw = f0[~np.isnan(f0)]
    row |= {"speaker": s, "path": p.name, "median_hz": float(np.median(raw)),
            "hz_std": float(raw.std())}
    rows.append(row)

print(f"\n{len(rows)}/{len(cache)} clips produced a feature row ({skipped} skipped), "
      f"{audio_s:.0f} s audio, F0 extraction {extract_s:.2f} s = {audio_s/extract_s:.0f}x realtime")

keys = ["st_std", "st_range", "st_slope_std", "st_dct1", "st_final_slope",
        "voiced_frac", "rms_db_std", "n_phrases"]
print("\nper-speaker means (ref_hz = pooled speaker reference):")
print(f"  {'spk':>6} {'ref_hz':>7} " + " ".join(f"{k:>13}" for k in keys))
for s in speakers:
    sub = [r for r in rows if r["speaker"] == s]
    print(f"  {s:>6} {refs[s]:7.1f} " + " ".join(f"{np.mean([r[k] for r in sub]):13.2f}" for k in keys))

def eta2(key):
    """Fraction of a feature's variance explained by speaker identity."""
    values = np.array([r[key] for r in rows]); groups = [r["speaker"] for r in rows]
    grand = values.mean()
    between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in
                  [values[np.array(groups) == s] for s in speakers])
    return between / ((values - grand) ** 2).sum()

print("\nvariance explained by speaker identity (eta^2) — the normalization check:")
for k in ["median_hz", "hz_std", "st_std", "st_range", "st_dct1", "st_final_slope", "rms_db_std"]:
    print(f"  {k:16s} {eta2(k):.3f}")

rows.sort(key=lambda r: r["st_range"])
print("\nleast expressive (lowest st_range):")
for r in rows[:3]:
    print(f"  {r['path']:36s} st_range {r['st_range']:5.2f}  st_std {r['st_std']:4.2f}")
print("most expressive (highest st_range):")
for r in rows[-3:]:
    print(f"  {r['path']:36s} st_range {r['st_range']:5.2f}  st_std {r['st_std']:4.2f}")

print("\n" + "=" * 78)
print("TEST C — LJSpeech: does st_final_slope separate questions from statements?")
print("=" * 78)
meta = [l.rstrip("\n").split("|") for l in open("data/ljspeech/metadata.csv")]
questions = [m[0] for m in meta if m[1].rstrip().endswith("?")]
statements = [m[0] for m in meta if m[1].rstrip().endswith(".")]
random.seed(2)
statements = random.sample(statements, len(questions))

def collect(names):
    out = []
    for name in names:
        path = pathlib.Path("data/ljspeech/audio") / name
        if not path.exists():
            continue
        y, sr = sf.read(path, dtype="float32")
        a = clip(y, sr)
        f0 = f0_contour(a)
        r = reference_pitch(f0)
        if r:
            out.append((a, f0, r))
    return out

q, d = collect(questions), collect(statements)
ref = speaker_reference([r for _, _, r in q + d])
qs = [f for f in (prosody_features(f0, a, ref) for a, f0, _ in q) if f]
ds = [f for f in (prosody_features(f0, a, ref) for a, f0, _ in d) if f]
print(f"\nspeaker reference {ref:.1f} Hz   |   {len(qs)} questions, {len(ds)} statements")
for k in ["st_final_slope", "st_dct1", "st_std", "st_range"]:
    a = np.array([r[k] for r in qs]); b = np.array([r[k] for r in ds])
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  {k:16s} question {a.mean():+7.3f} +/- {a.std():.3f}   "
          f"statement {b.mean():+7.3f} +/- {b.std():.3f}   Cohen d {(a.mean()-b.mean())/pooled:+.2f}")
rise_q = np.mean([r["st_final_slope"] > 0 for r in qs])
rise_d = np.mean([r["st_final_slope"] > 0 for r in ds])
print(f"  fraction with a rising final: questions {rise_q:.2f}  statements {rise_d:.2f}")
