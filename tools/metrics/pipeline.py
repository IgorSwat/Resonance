from dataclasses import dataclass
from typing import Any

from tools.metrics.clipping_ratio import ClippingRatioMetric
from tools.metrics.ctc_alignment_metric import CtcAlignmentMetric
from tools.metrics.effective_bandwidth import EffectiveBandwidthMetric
from tools.metrics.mains_hum import MainsHumMetric
from tools.metrics.metric import Metric
from tools.metrics.multi_speaker import MultiSpeakerMetric
from tools.metrics.nisqa import NisqaMetric
from tools.metrics.spectral_flatness import SpectralFlatnessMetric
from tools.metrics.types import QualityConfig, QualityVerdict


@dataclass
class Stage:
    verdict: QualityVerdict
    metric: Metric
    bounds: dict[str, Any]


class Pipeline:
    """
    Quality cascade: cheapest filter first, stopping at the first stage that rejects.

    DSP stages run before the model stages, so most rejects cost microseconds and the NISQA,
    segmentation and CTC forward passes only ever see audio that already survived. Ordering is the
    cascade of knowledge/data_filtering.md; the audio is accepted only if every stage passes.

    The CTC stage scores the transcript it is given, so audio without one cannot be verified
    and is rejected; switch it off with config.ctc_enabled when filtering untranscribed audio. Clips below config.min_duration are rejected before any stage runs: the
    metrics only warn about short audio, and every score they produce is unreliable there.
    """

    def __init__(self, config=None, verbose=False):
        self.config = config or QualityConfig()
        self.verbose = verbose
        self.stages = self._build(self.config)

    def run(self, audio, transcript=None):
        """
        The verdict for one clip: QualityVerdict.ACCEPTED, or the stage that rejected it.
        """

        duration = len(audio["audio"]) / audio["sample_rate"]
        if duration < self.config.min_duration:
            self._print(QualityVerdict.TOO_SHORT, audio, transcript)
            return QualityVerdict.TOO_SHORT
        if duration > self.config.max_duration:
            self._print(QualityVerdict.TOO_LONG, audio, transcript)
            return QualityVerdict.TOO_LONG

        for stage in self.stages:
            if stage.metric.validate(audio, transcript=transcript, **stage.bounds):
                continue
            self._print(stage.verdict, audio, transcript)
            return stage.verdict

        if self.verbose:
            print(f"ACCEPTED  ({len(self.stages)} stages passed)")
        return QualityVerdict.ACCEPTED

    @staticmethod
    def _build(config):
        stages = [
            Stage(
                QualityVerdict.EFFECTIVE_BANDWIDTH,
                EffectiveBandwidthMetric(
                    band_hz=config.bandwidth_band_hz, min_duration=config.min_duration,
                    cliff_db=config.bandwidth_cliff_db,
                ),
                {"lbound": config.bandwidth_min_frac, "rbound": 1.0},
            ),
            Stage(
                QualityVerdict.CLIPPING_RATIO,
                ClippingRatioMetric(),
                {"lbound": 0.0, "rbound": config.clipping_max_ratio},
            ),
            Stage(
                QualityVerdict.MAINS_HUM,
                MainsHumMetric(
                    freq_tolerance=config.hum_freq_tolerance, min_duration=config.min_duration
                ),
                {"rbound": config.hum_max_prominence},
            ),
            Stage(
                QualityVerdict.SPECTRAL_FLATNESS,
                SpectralFlatnessMetric(
                    loudness_threshold=config.flatness_loudness_threshold,
                    min_duration=config.min_duration,
                ),
                {"lbound": config.flatness_min, "rbound": config.flatness_max},
            ),
        ]
        # before NISQA, though both are model stages: segmentation costs 19 ms against NISQA's
        # 32 and rejects a larger share of what reaches it (38% against 19% on a batch of
        # Emilia EN), so it is the cheaper filter of the two
        if config.multi_speaker_enabled:
            stages.append(
                Stage(
                    QualityVerdict.MULTI_SPEAKER,
                    MultiSpeakerMetric(device=config.multi_speaker_device),
                    {"rbound": config.multi_speaker_max},
                )
            )
        stages.append(
            Stage(
                QualityVerdict.NISQA,
                NisqaMetric(
                    min_duration=config.min_duration, max_duration=config.nisqa_max_duration,
                    device=config.nisqa_device,
                ),
                {"lbound": config.nisqa_min},
            )
        )
        if config.ctc_enabled:
            stages.append(
                Stage(
                    QualityVerdict.CTC_ALIGNMENT,
                    CtcAlignmentMetric(device=config.ctc_device,
                                       uroman=config.ctc_uroman_enabled),
                    {"rbound": config.ctc_max},
                )
            )
        return stages

    def describe(self, verdict, audio, transcript=None):
        """
        Why a clip was rejected: every bounded score, and the bound it was held to.

        The scores are recomputed here rather than kept from validate(), so that an accepted
        clip never pays for diagnostics it does not need.
        """

        if verdict is QualityVerdict.ACCEPTED:
            return []
        if verdict in (QualityVerdict.TOO_SHORT, QualityVerdict.TOO_LONG):
            duration = len(audio["audio"]) / audio["sample_rate"]
            return [f"  {'duration':16} {duration:9.4f}   allowed "
                    f"{self.config.min_duration:.4f} .. {self.config.max_duration:.4f}"]

        stage = next(s for s in self.stages if s.verdict is verdict)
        try:
            scores = stage.metric.evaluate(audio, transcript=transcript)
        except Exception as error:
            return [f"  score unavailable: {error}"]
        if not isinstance(scores, dict):
            scores = {verdict.value: scores}
        return [f"  {name:16} {value:9.4f}   allowed {allowed}"
                for name, value in scores.items()
                if (allowed := _allowed(name, stage.bounds))]

    def _print(self, verdict, audio, transcript):
        if not self.verbose:
            return
        print(f"REJECTED by {verdict.value}")
        for line in self.describe(verdict, audio, transcript):
            print(line)


def _allowed(name, bounds):
    """
    The bound that applies to one score, as text — None when the score is unbounded.
    """

    lower, upper = bounds.get("lbound"), bounds.get("rbound")
    if isinstance(lower, dict) or isinstance(upper, dict):
        lower = lower.get(name) if isinstance(lower, dict) else None
        upper = upper.get(name) if isinstance(upper, dict) else None
        if lower is None and upper is None:
            return None
    if lower is not None and upper is not None:
        return f"{lower:.4f} .. {upper:.4f}"
    if lower is not None:
        return f">= {lower:.4f}"
    return f"<= {upper:.4f}" if upper is not None else None
