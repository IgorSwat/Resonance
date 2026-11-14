"""The contract of an audio codec that finalize can encode the dataset with."""

import abc
from typing import ClassVar

import numpy as np

from resonance.io.audio import Audio


class Codec(abc.ABC):
    """Waveform -> discrete tokens. Each codec writes to its own codecs/<name>/ folder of the dataset."""

    name: ClassVar[str]

    @abc.abstractmethod
    def encode(self, audio: Audio) -> np.ndarray:
        """One clip's tokens as an int16 array of shape (codebooks, frames)."""
