from typing import override

import numpy as np

from tools.metrics.metric import Metric


class EffectiveBandwidthMetric(Metric):
    """
    Fraction of Nyquist still carrying speech.

    A band is speech-bearing when its level moves over time; an artificial band limit leaves
    a dead band whose level never moves regardless of what is said. Comparing each band's
    temporal dynamic range against the speech band's makes the score independent of the
    file's noise floor and of the sample rate. See knowledge/data_filtering.md §1.1.

    Dynamics alone are not enough on lossy audio: a codec's sparse noise lines above its own
    lowpass fluctuate as much as speech does, so a 10 kHz mp3 would read as fullband. A band
    therefore also has to carry energy comparable to the band below it — a cliff steeper than
    cliff_db between successive bands is a codec's lowpass, not the speaker's voice, whose
    spectrum falls by a few dB per band. Comparing successive bands rather than the speech
    band keeps the test independent of natural spectral tilt.
    """

    def __init__(
        self,
        nfft=2048,
        hop=512,
        band_hz=1000.0,
        dynamics_ratio=0.4,
        start_hz=3000.0,
        speech_band=(1000.0, 4000.0),
        loud_percentile=40,
        min_duration=2.0,
        cliff_db=20.0,
    ):
        self.nfft = nfft
        self.hop = hop
        self.band_hz = band_hz
        self.dynamics_ratio = dynamics_ratio
        self.start_hz = start_hz
        self.speech_band = speech_band
        self.loud_percentile = loud_percentile
        self.min_duration = min_duration
        self.cliff_db = cliff_db

    @override
    def evaluate(self, audio, transcript=None):
        return self.report(audio)["frac"]

    @override
    def validate(self, audio, lbound, rbound, transcript=None):
        try:
            return lbound <= self.evaluate(audio) <= rbound
        except Exception as error:
            print(f"EffectiveBandwidthMetric failed: {error}")
            return False

    def report(self, audio):
        """
        Cutoff in Hz, fraction of Nyquist, and the band dynamics the cutoff was derived from.
        """

        y = np.asarray(audio["audio"], dtype=np.float64)
        sample_rate = audio["sample_rate"]
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < self.nfft:
            raise ValueError(f"Audio shorter than a single {self.nfft}-sample frame")
        if len(y) / sample_rate < self.min_duration:
            print(
                f"Warning: audio shorter than {self.min_duration} s"
                " — high-frequency content may be absent by chance"
            )

        dynamics = self._band_dynamics(y, sample_rate)
        speech = np.median(
            [span for upper, span, _ in dynamics if self.speech_band[0] <= upper <= self.speech_band[1]]
        )

        cutoff = self.start_hz
        reference = next(level for upper, _, level in dynamics if upper > self.start_hz)
        for upper, span, level in dynamics:
            if upper <= self.start_hz:
                continue
            # a band past the cutoff is either dead-flat or carries nothing but codec noise;
            # both read as no speech, whatever its fluctuation. The energy test is relative to
            # the last band that carried speech, so natural spectral tilt never trips it.
            # cliff_db=None restores the dynamics-only walk.
            if span < self.dynamics_ratio * speech or (
                    self.cliff_db is not None and level < reference - self.cliff_db):
                break
            cutoff = upper
            reference = level

        return {
            "cutoff": cutoff,
            "frac": cutoff / (sample_rate / 2),
            "speech_dynamics": float(speech),
        }

    def _band_dynamics(self, y, sample_rate):
        """
        Per-band temporal dynamic range in dB, over the loud half of the frames.
        """

        window = np.hanning(self.nfft)
        frames = 1 + (len(y) - self.nfft) // self.hop
        spectra = np.stack(
            [
                np.abs(np.fft.rfft(y[i * self.hop : i * self.hop + self.nfft] * window)) ** 2
                for i in range(frames)
            ]
        )
        energy = spectra.sum(axis=1)
        spectra = spectra[energy > np.percentile(energy, self.loud_percentile)]
        freqs = np.fft.rfftfreq(self.nfft, 1 / sample_rate)

        dynamics = []
        for lower in np.arange(0.0, sample_rate / 2, self.band_hz):
            upper = min(lower + self.band_hz, sample_rate / 2)
            if upper - lower < self.band_hz / 2:
                continue
            band = spectra[:, (freqs >= lower) & (freqs < upper)]
            if not band.size:
                continue
            level = 10 * np.log10(band.mean(axis=1) + 1e-20)
            dynamics.append(
                (float(upper), float(np.percentile(level, 95) - np.percentile(level, 20)),
                 float(10 * np.log10(band.mean() + 1e-20)))
            )
        return dynamics
