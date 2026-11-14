"""The quality cascade: cheapest metric first, stopping at the first one a clip fails.

DSP metrics run before the model metrics, so most rejects cost microseconds and the segmentation, NISQA and CTC
forward passes only see audio that already survived. The order is the config's `quality.metrics` list.
"""

from typing import Any

import numpy as np

from resonance.config.schema import QualityConfig
from resonance.io.audio import Audio
from resonance.metrics.metric import Metric, Unscorable

ACCEPTED, ERROR = "accepted", "error"
SILENCE, TOO_SHORT, TOO_LONG = "silence", "too_short", "too_long"
METRICS = ("bandwidth", "clipping", "hum", "flatness", "multi_speaker", "nisqa", "ctc")


def build(name: str, config: QualityConfig) -> tuple[Metric, Any]:
    """The metric called `name` and the bounds the config holds it to. Imported here, so a metric that is switched
    off never loads its libraries."""

    if name == "bandwidth":
        from resonance.metrics.effective_bandwidth import EffectiveBandwidthMetric
        return (EffectiveBandwidthMetric(band_hz=config.bandwidth.band_hz, cliff_db=config.bandwidth.cliff_db),
                config.bandwidth.min_frac)
    if name == "clipping":
        from resonance.metrics.clipping_ratio import ClippingRatioMetric
        return ClippingRatioMetric(), config.clipping.max_ratio
    if name == "hum":
        from resonance.metrics.mains_hum import MainsHumMetric
        return MainsHumMetric(freq_tolerance=config.hum.freq_tolerance), config.hum.max_prominence
    if name == "flatness":
        from resonance.metrics.spectral_flatness import SpectralFlatnessMetric
        return (SpectralFlatnessMetric(loudness_threshold=config.flatness.loudness_threshold),
                (config.flatness.min, config.flatness.max))
    if name == "multi_speaker":
        from resonance.metrics.multi_speaker import MultiSpeakerMetric
        return MultiSpeakerMetric(), config.multi_speaker.max
    if name == "nisqa":
        from resonance.metrics.nisqa import NisqaMetric
        return (NisqaMetric(min_duration=config.min_duration, max_duration=config.nisqa.max_duration),
                config.nisqa.min)
    if name == "ctc":
        from resonance.metrics.ctc_alignment import CtcAlignmentMetric
        return CtcAlignmentMetric(uroman=config.ctc.uroman), config.ctc.max
    raise KeyError(f"Unknown metric {name!r}; known: {', '.join(METRICS)}")


class Cascade:
    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self.stages = [(name, *build(name, config)) for name in config.metrics]

    def run(self, audio: Audio, transcript: str | None = None) -> tuple[str, dict[str, dict]]:
        """(verdict, scores): ACCEPTED or what rejected the clip, and every score computed on the way.

        The verdict of a rejection is the metric's name. A metric's exception is not caught here: it is an error,
        not a verdict, and the caller decides what to do with it.
        """

        y = audio["audio"]
        if y.size and np.ptp(y) == 0:
            return SILENCE, {}
        duration = len(y) / audio["sample_rate"]
        if duration < self.config.min_duration:
            return TOO_SHORT, {}
        if duration > self.config.max_duration:
            return TOO_LONG, {}

        scores: dict[str, dict] = {}
        for name, metric, bounds in self.stages:
            try:
                scores[name] = metric.evaluate(audio, transcript)
            except Unscorable as reason:
                scores[name] = {"unscorable": str(reason)}
                return name, scores
            if not metric.passes(scores[name], bounds):
                return name, scores
        return ACCEPTED, scores
