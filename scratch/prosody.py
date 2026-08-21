"""Prosody / F0 feature extraction, per knowledge/data_filtering.md section 2 (Level 2).

Two passes: pass 1 collects a per-speaker pitch reference, pass 2 turns each contour into a
fixed-width feature row expressed in semitones relative to that reference.
"""

import numpy as np
import parselmouth
from scipy.fft import dct

HOP = 0.010
PITCH_FLOOR = 60.0
PITCH_CEILING = 400.0
MAX_GAP_S = 0.20      # unvoiced gaps longer than this are phrase boundaries, not obstruents
FINAL_S = 0.30        # window used for the phrase-final slope
CONTOUR_POINTS = 64
DCT_COEFFS = 4
MIN_VOICED_FRAMES = 30
MIN_VOICED_FRAC = 0.25


def f0_contour(audio, floor=PITCH_FLOOR, ceiling=PITCH_CEILING, hop=HOP):
    """Per-frame F0 in Hz, NaN where unvoiced. Praat reports unvoiced as 0."""

    y = np.asarray(audio["audio"], dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sound = parselmouth.Sound(y, audio["sample_rate"])
    f0 = sound.to_pitch(time_step=hop, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = f0.selected_array["frequency"]
    return np.where(f0 > 0, f0, np.nan)


def reference_pitch(f0):
    """Pass 1: what each clip contributes to its speaker's pitch reference."""

    voiced = ~np.isnan(f0)
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None
    return {"median_hz": float(np.median(f0[voiced])), "n_voiced": int(voiced.sum())}


def speaker_reference(clips):
    """Voiced-frame-weighted mean of the per-clip medians. Pool all of a speaker's clips."""

    medians = np.array([c["median_hz"] for c in clips])
    weights = np.array([c["n_voiced"] for c in clips], dtype=float)
    return float(np.average(medians, weights=weights))


def prosody_features(f0, audio, ref_hz):
    """Pass 2: fixed-width prosody row for one clip."""

    voiced = ~np.isnan(f0)
    if voiced.sum() < MIN_VOICED_FRAMES or voiced.mean() < MIN_VOICED_FRAC:
        return None

    semitones = 12 * np.log2(f0 / ref_hz)
    observed = semitones[voiced]
    row = {
        "st_std": float(observed.std()),
        "st_range": float(np.percentile(observed, 95) - np.percentile(observed, 5)),
        "st_p95": float(np.percentile(observed, 95)),
        "st_slope_std": float(np.diff(observed).std()),
        "voiced_frac": float(voiced.mean()),
    }

    phrases = [p for p in split_phrases(semitones) if len(p) >= 10]
    if not phrases:
        return None

    shape = np.concatenate(phrases)
    grid = np.interp(
        np.linspace(0, 1, CONTOUR_POINTS), np.linspace(0, 1, len(shape)), shape
    )
    coefficients = dct(grid, type=2, norm="ortho")[1 : DCT_COEFFS + 1]
    row.update({f"st_dct{i}": float(c) for i, c in enumerate(coefficients, 1)})

    slopes = [final_slope(p) for p in phrases]
    row["st_final_slope"] = float(slopes[-1])
    row["st_phrase_slope_mean"] = float(np.mean(slopes))
    row["n_phrases"] = len(phrases)
    row.update(energy_features(audio))
    return row


def split_phrases(semitones):
    """Split at long unvoiced gaps; interpolate across short ones."""

    voiced = ~np.isnan(semitones)
    if not voiced.any():
        return []
    max_gap = int(round(MAX_GAP_S / HOP))
    edges = np.flatnonzero(np.diff(np.r_[1, voiced.astype(np.int8), 1]))
    gaps = edges.reshape(-1, 2)
    boundaries = [0] + [int(b) for a, b in gaps if b - a > max_gap] + [len(semitones)]

    phrases = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = semitones[start:stop]
        present = np.flatnonzero(~np.isnan(segment))
        if len(present) < 10:
            continue
        segment = segment[present[0] : present[-1] + 1]
        index = np.arange(len(segment))
        known = ~np.isnan(segment)
        phrases.append(np.interp(index, index[known], segment[known]))
    return phrases


def final_slope(phrase):
    """Semitones per second over the last FINAL_S of a phrase."""

    tail = phrase[-int(round(FINAL_S / HOP)) :]
    if len(tail) < 5:
        return 0.0
    return float(np.polyfit(np.arange(len(tail)) * HOP, tail, 1)[0])


def energy_features(audio, hop=HOP):
    """Loudness dynamics, measured on the frames that carry speech."""

    y = np.asarray(audio["audio"], dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    step = int(round(hop * audio["sample_rate"]))
    window = 2 * step
    n = max(0, (len(y) - window) // step + 1)
    rms = np.array([np.sqrt(np.mean(y[i * step : i * step + window] ** 2)) for i in range(n)])
    level = 20 * np.log10(rms + 1e-10)
    speech = level > np.percentile(level, 95) - 40
    if speech.sum() < 10:
        return {"rms_db_std": 0.0, "rms_db_range": 0.0}
    level = level[speech]
    return {
        "rms_db_std": float(level.std()),
        "rms_db_range": float(np.percentile(level, 95) - np.percentile(level, 20)),
    }
