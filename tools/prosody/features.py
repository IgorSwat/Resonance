import numpy as np
from scipy.fft import dct

from tools.prosody.constants import (
    CONTOUR_POINTS,
    DCT_COEFFS,
    FINAL_S,
    HOP,
    MIN_VOICED_FRAC,
    MIN_VOICED_FRAMES,
)
from tools.prosody.contour import split_phrases


def prosody_features(semitones, duration):
    """
    Fixed-width prosody row for one clip, from its speaker-normalized contour.

    Three blocks: how much pitch moves (st_range, st_slope_std), what shape it traces (DCT
    coefficients of the phrase-concatenated contour on a fixed grid, plus the phrase-final
    slope, which carries the question/statement distinction), and how the clip is phrased.
    Returns None when the clip carries too little voiced speech to describe.
    """

    voiced = ~np.isnan(semitones)
    if voiced.sum() < MIN_VOICED_FRAMES or voiced.mean() < MIN_VOICED_FRAC:
        return None
    phrases = split_phrases(semitones)
    if not phrases:
        return None

    observed = semitones[voiced]
    shape = np.concatenate(phrases)
    grid = np.interp(
        np.linspace(0, 1, CONTOUR_POINTS), np.linspace(0, 1, len(shape)), shape
    )
    # the resample makes clips of different length comparable; coefficient 0 is the overall
    # level, i.e. speaker identity, so the shape block starts at 1
    coefficients = dct(grid, type=2, norm="ortho")[1 : DCT_COEFFS + 1]

    row = {
        # percentiles rather than min/max, so one bad frame cannot define the range
        "st_range": float(np.percentile(observed, 95) - np.percentile(observed, 5)),
        "st_slope_std": float(np.diff(observed).std()),
        "st_final_slope": final_slope(phrases[-1]),
        "voiced_frac": float(voiced.mean()),
        "voiced_onsets_per_s": onset_rate(voiced, duration),
        "n_phrases": len(phrases),
    }
    row.update({f"st_dct{i}": float(c) for i, c in enumerate(coefficients, 1)})
    return row


def final_slope(phrase):
    """Semitones per second over the last FINAL_S of a phrase: rising vs falling final."""

    tail = phrase[-int(round(FINAL_S / HOP)) :]
    if len(tail) < 5:
        return 0.0
    return float(np.polyfit(np.arange(len(tail)) * HOP, tail, 1)[0])


def onset_rate(voiced, duration):
    """Voiced-group onsets per second — a speaking-rate proxy, standing in for a CTC aligner."""

    return float((np.diff(np.r_[0, voiced.astype(np.int8)]) == 1).sum() / duration)
