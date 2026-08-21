"""Scan LibriTTS with the prosody pipeline and copy the least/most expressive clips to ./tmp.

Applies the three fixes found during validation: speaker-adapted pitch range (octave errors),
a 2 s minimum (short clips inflate the tail), and a per-clip octave guard against the reference.
"""

import collections, pathlib, random, shutil, time
import numpy as np, soundfile as sf
from prosody import f0_contour, reference_pitch, prosody_features

N_SPEAKERS, N_CLIPS, N_PICK = 250, 12, 12
MIN_DURATION = 2.0
OCTAVE_GUARD = (0.67, 1.5)      # clip median must sit within this of the speaker reference
OUT = pathlib.Path("tmp")

random.seed(7)
by_speaker = collections.defaultdict(list)
for p in sorted(pathlib.Path("data/librispeech/audio").iterdir()):
    by_speaker[p.name.split("_")[0]].append(p)
speakers = sorted(random.sample([s for s, v in by_speaker.items() if len(v) >= N_CLIPS], N_SPEAKERS))

t0 = time.time()
clips, audio_s = [], 0.0
for s in speakers:
    for p in random.sample(by_speaker[s], N_CLIPS):
        y, sr = sf.read(p, dtype="float32")
        if len(y) / sr < MIN_DURATION:
            continue
        audio_s += len(y) / sr
        audio = {"audio": y.astype(np.float64), "sample_rate": sr}
        f0 = f0_contour(audio)
        clips.append((s, p, audio, reference_pitch(f0)))
print(f"pass 1: {len(clips)} clips >= {MIN_DURATION}s, {audio_s/60:.0f} min audio, {time.time()-t0:.0f}s")

# robust reference: median of clip medians, so octave-doubled clips cannot drag it up
refs = {}
for s in speakers:
    medians = [c[3]["median_hz"] for c in clips if c[0] == s and c[3]]
    if medians:
        refs[s] = float(np.median(medians))

t0 = time.time()
rows, rejected = [], collections.Counter()
for s, path, audio, r in clips:
    if r is None or s not in refs:
        rejected["no reference"] += 1; continue
    ref = refs[s]
    if not OCTAVE_GUARD[0] * ref <= r["median_hz"] <= OCTAVE_GUARD[1] * ref:
        rejected["octave guard"] += 1; continue
    f0 = f0_contour(audio, floor=max(50.0, 0.55 * ref), ceiling=min(500.0, 1.9 * ref))
    row = prosody_features(f0, audio, ref)
    if row is None:
        rejected["no features"] += 1; continue
    row |= {"speaker": s, "path": path, "ref_hz": ref,
            "dur": len(audio["audio"]) / audio["sample_rate"]}
    rows.append(row)
print(f"pass 2 (adapted range): {len(rows)} rows, {dict(rejected)}, {time.time()-t0:.0f}s")

rows.sort(key=lambda r: r["st_range"])
values = np.array([r["st_range"] for r in rows])
print(f"\nst_range: mean {values.mean():.2f}  sd {values.std():.2f}  "
      f"p5 {np.percentile(values,5):.2f}  p95 {np.percentile(values,95):.2f}  "
      f"min {values.min():.2f}  max {values.max():.2f}")

for f in OUT.iterdir():
    if f.is_file():
        f.unlink()
print(f"cleared {OUT}/")

def emit(row, tag):
    name = (f"libritts_{tag}_range{row['st_range']:05.2f}_std{row['st_std']:04.2f}"
            f"_dct1{row['st_dct1']:+06.2f}_fslope{row['st_final_slope']:+06.1f}"
            f"_ref{row['ref_hz']:03.0f}hz_{row['dur']:.1f}s_{row['path'].name}")
    shutil.copy2(row["path"], OUT / name)
    return name

print(f"\n--- least expressive ({N_PICK}) ---")
for r in rows[:N_PICK]:
    print("  " + emit(r, "least-expr"))
print(f"\n--- most expressive ({N_PICK}) ---")
for r in reversed(rows[-N_PICK:]):
    print("  " + emit(r, "most-expr"))

print(f"\nspeakers in the least set: {len({r['speaker'] for r in rows[:N_PICK]})}/{N_PICK}"
      f" | in the most set: {len({r['speaker'] for r in rows[-N_PICK:]})}/{N_PICK}")
