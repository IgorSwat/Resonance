"""The contract every quality metric fulfils."""

import abc
from typing import Any

from resonance.io.audio import Audio


class Unscorable(ValueError):
    """
    The clip cannot be scored for a reason that is the clip's own (too short, silent, nothing to align).

    The cascade counts it as a rejection by that metric. Any other exception is an error, not a verdict.
    """


class Metric(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, audio: Audio, transcript: str | None = None) -> dict[str, float]:
        """The metric's scores for one clip."""

    @abc.abstractmethod
    def passes(self, scores: dict[str, float], bounds: Any) -> bool:
        """Whether the scores lie within the bounds the config gives this metric."""


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
