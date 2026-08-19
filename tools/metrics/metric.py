from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Common contract for audio quality metrics.

    Audio is given in the form produced by the loaders: a dict with an 'audio' waveform
    and a 'sample_rate'.
    """

    @abstractmethod
    def evaluate(self, audio, transcript=None):
        """Return the metric's score for the audio."""

    @abstractmethod
    def validate(self, audio, lbound, rbound, transcript=None):
        """Return whether the score lies within [lbound, rbound]; on any error, reject."""
