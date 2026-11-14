from typing import override

import numpy as np

from resonance.io.audio import Audio
from resonance.metrics.metric import Metric, Unscorable


class ClippingRatioMetric(Metric):
    """
    Fraction of samples lost to clipping.

    Counted over runs rather than individual samples: a peak-normalised file legitimately
    touches full scale for a sample or two, whereas clipping flattens the waveform, so only
    runs of at least min_run consecutive samples above the threshold are counted. See
    knowledge/data_filtering.md §1 (clipping rate).
    """

    def __init__(self, threshold: float = 0.99, min_run: int = 3) -> None:
        self.threshold = threshold
        self.min_run = min_run

    @override
    def passes(self, scores: dict[str, float], bounds: float) -> bool:
        """`bounds` is the highest acceptable clipped-sample ratio."""

        return scores["ratio"] <= bounds

    @override
    def evaluate(self, audio: Audio, transcript: str | None = None) -> dict[str, float]:
        """Clipped-sample ratio over runs, the raw per-sample ratio, and the longest run."""

        y = np.asarray(audio["audio"], dtype=np.float64)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if not y.size:
            raise Unscorable("Empty audio")

        clipped = np.abs(y) > self.threshold
        edges = np.diff(np.flatnonzero(np.diff(np.r_[0, clipped.view(np.int8), 0])))[::2]
        runs = edges[edges >= self.min_run]
        return {
            "ratio": float(runs.sum() / len(y)),
            "sample_ratio": float(clipped.mean()),
            "longest_run": int(edges.max()) if len(edges) else 0,
        }
