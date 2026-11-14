"""Scores compared only with clips of their own length."""

import numpy as np

# Wide enough that every band holds hundreds of clips in a 10k subset, narrow enough that the median similarity
# is flat inside one.
BANDS = ((0, 6), (6, 9), (9, 13), (13, 18), (18, float("inf")))


def banded_z(scores: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Each score as a z within its duration band. A band of one clip has no spread and is left at z = 0."""

    z = np.zeros_like(scores, dtype=float)
    for low, high in BANDS:
        band = (durations >= low) & (durations < high)
        if band.sum() > 1:
            z[band] = (scores[band] - scores[band].mean()) / scores[band].std()
    return z
