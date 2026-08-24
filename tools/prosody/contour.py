import numpy as np
import parselmouth

from tools.prosody.constants import (
    HOP,
    MAX_GAP_S,
    MIN_PHRASE_FRAMES,
    PITCH_CEILING,
    PITCH_FLOOR,
)


def f0_contour(audio, floor=PITCH_FLOOR, ceiling=PITCH_CEILING, hop=HOP):
    """
    Per-frame F0 in Hz, NaN where unvoiced.

    Praat's autocorrelation tracker, which reports unvoiced frames as 0. Every numpy reduction
    over the result is NaN unless the caller masks first. See knowledge/data_filtering.md §2
    for why this tracker rather than pyin or harvest.
    """

    y = np.asarray(audio["audio"], dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sound = parselmouth.Sound(y, audio["sample_rate"])
    pitch = sound.to_pitch(time_step=hop, pitch_floor=floor, pitch_ceiling=ceiling)
    frequency = pitch.selected_array["frequency"]
    return np.where(frequency > 0, frequency, np.nan)


def split_phrases(contour):
    """
    Split the contour at long unvoiced gaps, interpolating across the short ones.

    Gaps below MAX_GAP_S are obstruents. Interpolating across a real pause instead would
    fabricate a contour nobody spoke and pollute every shape feature derived from it.
    """

    voiced = ~np.isnan(contour)
    if not voiced.any():
        return []

    max_gap = int(round(MAX_GAP_S / HOP))
    edges = np.flatnonzero(np.diff(np.r_[1, voiced.astype(np.int8), 1]))
    gaps = edges.reshape(-1, 2)
    boundaries = [0] + [int(stop) for start, stop in gaps if stop - start > max_gap]
    boundaries.append(len(contour))

    phrases = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = contour[start:stop]
        present = np.flatnonzero(~np.isnan(segment))
        if len(present) < MIN_PHRASE_FRAMES:
            continue
        segment = segment[present[0] : present[-1] + 1]
        index = np.arange(len(segment))
        known = ~np.isnan(segment)
        phrases.append(np.interp(index, index[known], segment[known]))
    return phrases
