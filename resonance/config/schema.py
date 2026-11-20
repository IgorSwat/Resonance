"""
The pipeline configuration: one dataclass per section of configs/<pipeline>.yaml.

Defaults are the values calibrated for YODAS2-Sidon (knowledge/yodas2_sidon.md, knowledge/data_filtering.md).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    name: str = ""                         # set per chunk by scripts/run_language.py, or with --set for one run
    data_root: Path = Path("data/yodas2_sidon")


@dataclass
class BatchRange:
    shard: str
    first: int
    last: int


@dataclass
class SourceConfig:
    language: str = ""
    batches: list[BatchRange] = field(default_factory=list)


@dataclass
class ResourcesConfig:
    gpus: str | list[int] = "all"
    fetch_ahead: int = 3
    cpu_workers: int = 24
    threads_per_worker: int = 4


@dataclass
class DiskConfig:
    raw_gb_per_batch: float = 3.0
    clips_gb_per_batch: float = 3.5
    kept_share: float = 0.5


@dataclass
class SegmentConfig:
    whisper: str = "large-v3-turbo"
    workers: int = 3
    batch_size: int = 16
    beam_size: int = 5
    margin: float = 2.0
    target_duration: float = 8.0
    min_duration: float = 3.0
    max_duration: float = 30.0
    max_pause: float = 2.0
    min_gap: float = 0.15
    lead: float = 0.1
    trail: float = 0.2
    keep_raw: bool = False


@dataclass
class BandwidthConfig:
    min_frac: float = 0.85
    band_hz: float = 1000.0
    cliff_db: float = 20.0


@dataclass
class ClippingConfig:
    max_ratio: float = 0.0001


@dataclass
class HumConfig:
    max_prominence: float = 15.0
    freq_tolerance: float = 0.3


@dataclass
class FlatnessConfig:
    min: float = 0.0005
    max: float = 0.25
    loudness_threshold: float = 35.0


@dataclass
class MultiSpeakerConfig:
    max: dict[str, float] = field(default_factory=lambda: {"second": 0.0, "overlap": 0.0})


@dataclass
class NisqaConfig:
    min: dict[str, float] = field(default_factory=lambda: {
        "noisiness": 3.0, "discontinuity": 3.5, "coloration": 3.25, "loudness": 3.0})
    max_duration: float = 50.0


@dataclass
class CtcConfig:
    uroman: bool = True
    max: dict[str, float] = field(default_factory=lambda: {
        "loss": 0.7, "max_loss": 1.0, "maxrate": 30.0, "gap_speech": 0.06, "trail_speech": 0.05})


@dataclass
class QualityConfig:
    min_duration: float = 3.0
    max_duration: float = 30.0
    metrics: list[str] = field(default_factory=lambda: [
        "bandwidth", "clipping", "hum", "flatness", "multi_speaker", "nisqa", "ctc"])
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    hum: HumConfig = field(default_factory=HumConfig)
    flatness: FlatnessConfig = field(default_factory=FlatnessConfig)
    multi_speaker: MultiSpeakerConfig = field(default_factory=MultiSpeakerConfig)
    nisqa: NisqaConfig = field(default_factory=NisqaConfig)
    ctc: CtcConfig = field(default_factory=CtcConfig)
    workers: int = 5
    max_error_rate: float = 0.01


@dataclass
class SourcePassConfig:
    enabled: bool = True
    max_flag_rate: float = 0.3
    min_clips: int = 3


@dataclass
class SpeakerDriftConfig:
    enabled: bool = True
    max_z: float = -2.0
    workers: int = 4


@dataclass
class TranscriptFixupConfig:
    whisper: str = "large-v3-turbo"
    beam_size: int = 5
    workers: int = 3
    normalize: bool = True
    trim: bool = True
    terms: bool = True


@dataclass
class SpeakerFixupConfig:
    threshold: float = 0.5
    linkage: str = "single"
    clips_per_speaker: int = 4


@dataclass
class SelectConfig:
    min_per_speaker: int = 2
    max_per_speaker: int = 64
    max_minutes_per_speaker: float | None = None
    seed: int = 0


@dataclass
class HighlightsConfig:
    per_list: int = 50
    per_speaker: int = 2
    workers: int = 1


@dataclass
class FinalizeConfig:
    move: bool = True
    zip: bool = True
    compression: str = "deflate"
    codecs: list[str] = field(default_factory=list)


@dataclass
class UploadConfig:
    remote: str = ""                       # rclone remote name; empty uploads to a local path (tests)
    root_folder_id: str = ""               # Drive folder the path is relative to
    path: str = "train/{language_name}/yodas2_sidon/{run}"
    highlights: bool = True
    transfers: int = 4
    retries: int = 3
    stats_seconds: int = 30                # interval of the upload progress lines
    quota_wait_minutes: float = 60.0       # wait before retrying once Drive's daily upload limit is reached


@dataclass
class OrchestrateConfig:
    chunk_batches: int = 50                # chunks are aligned to multiples of this within a shard
    shards: list[str] = field(default_factory=list)     # e.g. ["004", "005"]; empty takes every shard
    poll_seconds: float = 60.0
    stall_minutes: float = 30.0            # run.log silent this long while the pipeline lives: it hangs
    max_failures: int = 3                  # a batch failing this often is quarantined; a pool stage stops everything
    min_free_gb: float = 130.0             # a chunk's GPU part starts only with this much free disk
    upload: UploadConfig = field(default_factory=UploadConfig)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    disk: DiskConfig = field(default_factory=DiskConfig)
    pipeline: list[str] = field(default_factory=lambda: [
        "fetch", "segment", "quality", "source_pass", "speaker_drift", "discard_rejected",
        "merge", "transcript_fixup", "speaker_fixup", "select", "highlights", "finalize"])
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    source_pass: SourcePassConfig = field(default_factory=SourcePassConfig)
    speaker_drift: SpeakerDriftConfig = field(default_factory=SpeakerDriftConfig)
    transcript_fixup: TranscriptFixupConfig = field(default_factory=TranscriptFixupConfig)
    speaker_fixup: SpeakerFixupConfig = field(default_factory=SpeakerFixupConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)
    finalize: FinalizeConfig = field(default_factory=FinalizeConfig)
    orchestrate: OrchestrateConfig = field(default_factory=OrchestrateConfig)
