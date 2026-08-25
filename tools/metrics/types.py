from dataclasses import dataclass, field, fields
from enum import Enum

from tools.metrics.ctc_alignment_metric import DEFAULT_RBOUND as CTC_RBOUND
from tools.metrics.multi_speaker import DEFAULT_RBOUND as SPEAKER_RBOUND
from tools.metrics.nisqa import DEFAULT_LBOUND as NISQA_LBOUND


class QualityVerdict(Enum):
    """
    Outcome of the quality pipeline: accepted, or the stage that rejected the audio.
    """

    ACCEPTED = "accepted"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    EFFECTIVE_BANDWIDTH = "effective_bandwidth"
    CLIPPING_RATIO = "clipping_ratio"
    MAINS_HUM = "mains_hum"
    SPECTRAL_FLATNESS = "spectral_flatness"
    NISQA = "nisqa"
    MULTI_SPEAKER = "multi_speaker"
    CTC_ALIGNMENT = "ctc_alignment"

    @property
    def accepted(self):
        return self is QualityVerdict.ACCEPTED


@dataclass
class QualityConfig:
    """
    Bounds and hyperparameters for every stage of the pipeline.

    Defaults are the ones each metric was calibrated with — see knowledge/data_filtering.md.
    They are starting points: the note advises percentile gates per source corpus, since
    Common Voice, parliamentary audio and audiobooks have very different baselines.
    """

    bandwidth_min_frac: float = 0.85
    bandwidth_band_hz: float = 1000.0

    clipping_max_ratio: float = 1e-4

    hum_max_prominence: float = 15.0
    hum_freq_tolerance: float = 0.3

    flatness_min: float = 0.0005
    flatness_max: float = 0.25
    flatness_loudness_threshold: float = 35.0

    # the metric's own defaults gate discontinuity and coloration; mos and loudness are added
    # here as liberal backstops, since both move with content alone and cannot be gated tightly
    nisqa_min: dict = field(
        default_factory=lambda: {**NISQA_LBOUND, "mos": 3.5, "noisiness": 3.5, "loudness": 3.0}
    )
    nisqa_max_duration: float = 50.0

    multi_speaker_enabled: bool = True
    multi_speaker_max: dict = field(default_factory=lambda: dict(SPEAKER_RBOUND))
    multi_speaker_device: str | None = None

    ctc_enabled: bool = True
    ctc_max: dict = field(default_factory=lambda: dict(CTC_RBOUND))
    ctc_device: str | None = None

    min_duration: float = 5.0
    max_duration: float = 30.0

    @classmethod
    def from_yaml(cls, path):
        """
        Load a config from a flat YAML mapping; absent keys keep their default.
        """

        import yaml

        with open(path) as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise TypeError(f"{path} must contain a mapping, got {type(values).__name__}")

        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}; expected {sorted(known)}")
        return cls(**values)
