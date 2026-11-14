"""What flows between stages, and where everything lives on disk."""

import dataclasses
import re
from pathlib import Path

from resonance.config.schema import Config


@dataclasses.dataclass(frozen=True)
class BatchRef:
    """One train-XXXXX batch of a language shard, e.g. en001/train-00091."""

    language: str
    shard: str
    index: int

    @property
    def name(self) -> str:
        return f"train-{self.index:05d}"

    @property
    def key(self) -> str:
        return f"{self.language}{self.shard}/{self.name}"

    @classmethod
    def parse(cls, text: str) -> "BatchRef":
        match = re.fullmatch(r"([a-z]+)(\d+)/(?:train-)?(\d+)", text)
        if not match:
            raise ValueError(f"Not a batch: {text!r}; expected e.g. en001/91")
        return cls(match[1], match[2], int(match[3]))


@dataclasses.dataclass(frozen=True)
class RawBatch:
    """A directory of <n>.flac beside <n>.metadata.json, one pair per video."""

    path: Path


@dataclasses.dataclass(frozen=True)
class ClipManifest:
    """A manifest CSV (resonance/io/manifest.py); its audio_path column is relative to the data root."""

    path: Path


@dataclasses.dataclass(frozen=True)
class Dataset:
    """The final product: <run>.csv, audio.zip (or audio/) and optional codecs/<codec>/."""

    path: Path


class Layout:
    """Every path of a run, derived from run.data_root."""

    def __init__(self, config: Config):
        self.root = config.run.data_root
        self.run_dir = self.root / "runs" / config.run.name
        self.dataset = self.run_dir / "dataset"

    def raw(self, batch: BatchRef) -> Path:
        return self.root / "raw" / batch.language / batch.shard / batch.name

    def clips(self, batch: BatchRef) -> Path:
        return self.root / "clips" / f"{batch.language}{batch.shard}" / batch.name

    def batch_dir(self, batch: BatchRef) -> Path:
        return self.root / "batches" / f"{batch.language}{batch.shard}" / batch.name


def batches(config: Config) -> list[BatchRef]:
    """Every batch the config's source names, in order."""

    return [BatchRef(config.source.language, span.shard, index)
            for span in config.source.batches for index in range(span.first, span.last + 1)]
