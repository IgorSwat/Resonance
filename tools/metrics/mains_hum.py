from typing import override

import numpy as np

from tools.metrics.metric import Metric


class MainsHumMetric(Metric):
    """
    Strength of mains hum, in dB above the surrounding noise floor.

    A recording is only rejected when the line also sits within freq_tolerance of 50 or
    60 Hz. Grid frequency is regulated to a few tens of millihertz and nothing else in audio
    holds a frequency that precisely, so the lock — not the prominence — is what separates
    hum from ordinary tonal noise. See knowledge/data_filtering.md §1.0.
    """

    def __init__(
        self,
        nfft=32768,
        harmonics=3,
        line_frequencies=(50.0, 60.0),
        freq_tolerance=0.3,
        search_hz=2.5,
        floor_band=(10.0, 30.0),
        min_duration=2.0,
    ):
        self.nfft = nfft
        self.harmonics = harmonics
        self.line_frequencies = line_frequencies
        self.freq_tolerance = freq_tolerance
        self.search_hz = search_hz
        self.floor_band = floor_band
        self.min_duration = min_duration

    @override
    def evaluate(self, audio, transcript=None):
        return self.report(audio)["prominence"]

    @override
    def validate(self, audio, lbound=-np.inf, rbound=10.0, transcript=None):
        try:
            report = self.report(audio)
        except Exception as error:
            print(f"HumMetric failed: {error}")
            return False
        if report["prominence"] > rbound and report["deviation"] <= self.freq_tolerance:
            return False
        return report["prominence"] >= lbound

    def report(self, audio):
        """Prominence of the strongest mains line in dB, its frequency, and its offset from the grid."""

        y = np.asarray(audio["audio"], dtype=np.float64)
        sample_rate = audio["sample_rate"]
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < self.nfft:
            raise ValueError(f"Audio shorter than a single {self.nfft}-sample window")
        if len(y) / sample_rate < self.min_duration:
            print(f"Warning: audio shorter than {self.min_duration} s — the line estimate is noisy")
        y = y - y.mean()

        window = np.hanning(self.nfft)
        step = self.nfft // 2
        spectra = np.stack(
            [
                np.abs(np.fft.rfft(y[i : i + self.nfft] * window)) ** 2
                for i in range(0, len(y) - self.nfft + 1, step)
            ]
        )
        levels = 10 * np.log10(np.median(spectra, axis=0) + 1e-30)
        freqs = np.fft.rfftfreq(self.nfft, 1 / sample_rate)

        best = max(
            (self._score_line(levels, freqs, line) for line in self.line_frequencies),
            key=lambda report: report["prominence"],
        )
        return best

    def _score_line(self, levels, freqs, line):
        prominences, fundamental = [], line
        for harmonic in range(1, self.harmonics + 1):
            centre = line * harmonic
            near = np.flatnonzero(np.abs(freqs - centre) <= self.search_hz)
            floor = levels[
                (np.abs(freqs - centre) >= self.floor_band[0])
                & (np.abs(freqs - centre) <= self.floor_band[1])
            ]
            if not len(near) or not len(floor):
                continue
            peak = near[np.argmax(levels[near])]
            prominences.append(levels[peak] - np.median(floor))
            if harmonic == 1:
                fundamental = self._interpolate(levels, freqs, peak)
        if not prominences:
            return {"prominence": -np.inf, "frequency": line, "deviation": np.inf, "line": line}
        return {
            "prominence": float(np.mean(prominences)),
            "frequency": float(fundamental),
            "deviation": float(abs(fundamental - line)),
            "line": line,
        }

    @staticmethod
    def _interpolate(levels, freqs, peak):
        """Parabolic fit around the peak bin — the line rarely falls on a bin centre."""

        if peak == 0 or peak + 1 >= len(levels):
            return freqs[peak]
        left, centre, right = levels[peak - 1], levels[peak], levels[peak + 1]
        shift = 0.5 * (left - right) / (left - 2 * centre + right + 1e-12)
        return freqs[peak] + shift * (freqs[1] - freqs[0])
