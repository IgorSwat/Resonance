from typing import override

import numpy as np

from resonance.io.audio import Audio
from resonance.metrics.metric import Metric, Unscorable


class SpectralFlatnessMetric(Metric):
    """
    Spectral flatness of the noise floor.

    Measured on the quiet (non-speech) frames only, so the score describes the background
    rather than the voice: near 1 means broadband hiss, near 0 means a tonal background —
    mains hum, music, or a whine. See knowledge/data_filtering.md §1.0.

    Flatness ignores level by construction, so a quiet dither floor scores as high as audible
    hiss. Both bounds are therefore only enforced when the floor is close enough to speech to
    be heard, i.e. when speech rises less than loudness_threshold dB (A-weighted) above it.
    That gap is A-weighted, since an unweighted one is dominated by low-frequency rumble that
    is inaudible.
    """

    def __init__(
        self,
        nfft: int = 2048,
        hop: int = 512,
        quiet_percentile: int = 10,
        min_hz: float = 20.0,
        loudness_threshold: float = 35.0,
    ) -> None:
        self.nfft = nfft
        self.hop = hop
        self.quiet_percentile = quiet_percentile
        self.min_hz = min_hz
        self.loudness_threshold = loudness_threshold

    @override
    def passes(self, scores: dict[str, float], bounds: tuple[float, float]) -> bool:
        """`bounds` is (lowest, highest) flatness, enforced only when the floor is loud enough to be heard."""

        lowest, highest = bounds
        if scores["flatness"] < lowest and scores["gap"] < self.loudness_threshold:
            return False
        return scores["flatness"] <= highest or scores["gap"] >= self.loudness_threshold

    @override
    def evaluate(self, audio: Audio, transcript: str | None = None) -> dict[str, float]:
        """Flatness of the noise floor, and how far speech rises above that floor in dB."""

        y = np.asarray(audio["audio"], dtype=np.float64)
        sample_rate = audio["sample_rate"]
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < self.nfft:
            raise Unscorable(f"Audio shorter than a single {self.nfft}-sample frame")
        y = y - y.mean()  # a DC offset leaks into the lowest bins and inflates the noise floor

        window = np.hanning(self.nfft)
        frames = 1 + (len(y) - self.nfft) // self.hop
        spectra = np.stack(
            [
                np.abs(np.fft.rfft(y[i * self.hop : i * self.hop + self.nfft] * window)) ** 2
                for i in range(frames)
            ]
        )
        freqs = np.fft.rfftfreq(self.nfft, 1 / sample_rate)
        levels = 10 * np.log10((spectra * _a_weighting(freqs)).sum(axis=1) + 1e-20)
        quiet = levels <= np.percentile(levels, self.quiet_percentile)

        floor = spectra[quiet][:, freqs >= self.min_hz] + 1e-20
        if not floor.size:
            raise Unscorable("No spectral content above min_hz in the quiet frames")

        flatness = np.exp(np.log(floor).mean(axis=1)) / floor.mean(axis=1)
        return {
            "flatness": float(np.median(flatness)),
            "gap": float(np.percentile(levels, 90) - np.median(levels[quiet])),
        }


def _a_weighting(freqs: np.ndarray) -> np.ndarray:
    """
    Power-domain A-weighting curve, approximating what the ear actually hears.
    """

    f2 = np.maximum(freqs, 1.0) ** 2
    response = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2)
    )
    return response**2
