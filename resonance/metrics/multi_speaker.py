from typing import override

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.audio.utils.powerset import Powerset

from resonance.io.audio import Audio
from resonance.metrics.metric import Metric, default_device

CHECKPOINT = "pyannote/segmentation-3.0"
SAMPLE_RATE = 16000
FIELDS = ("second", "overlap")
RESAMPLE = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}


class MultiSpeakerMetric(Metric):
    """
    Speech from anyone other than the intended speaker, in seconds (pyannote segmentation 3.0).

    Emilia and other diarised in-the-wild corpora label one speaker per clip, but the
    diarisation leaks: roughly a third of an Emilia batch has a second voice audible. Nothing
    else in the cascade sees it, because every other stage scores channel properties that a
    two-person exchange on one microphone passes cleanly.

    The model emits a frame-level powerset distribution over three local speakers, at most two
    of them simultaneous. Two scores are read off it, and they mean different things:

    - **second** — how long the second-most-active speaker talks, overlapping or not. Turn
      taking counts, so this is the recall-first score.
    - **overlap** — how long two speakers talk at once. A strict subset of the same evidence,
      and the confident one: every clip with any overlap was a confirmed second speaker.

    Scores are aggregated per window rather than stitched into one timeline: the model's
    speaker slots are local to each window, so only within-window quantities are comparable.
    """

    def __init__(self, device: str | None = None, step: float = 2.5) -> None:
        self.device = device or default_device()
        self.step = step
        self._model = None
        self._powerset = None

    @override
    def evaluate(self, audio: Audio, transcript: str | None = None) -> dict[str, float]:
        """
        Seconds of second-speaker speech, and of overlapped speech; higher is worse on both.
        """

        activity = self._activity(audio)
        frame = self._model.specifications.duration / activity.shape[1]
        speaking = np.sort(activity.sum(axis=1), axis=1) * frame
        return {
            "second": float(speaking[:, -2].max()),
            "overlap": float((activity.sum(axis=2) >= 2).sum(axis=1).max() * frame),
        }

    @override
    def passes(self, scores: dict[str, float], bounds: dict[str, float]) -> bool:
        """`bounds` maps a score to the most seconds allowed of it."""

        unknown = set(bounds) - set(FIELDS)
        if unknown:
            raise KeyError(f"Unknown speaker scores: {sorted(unknown)}; expected {list(FIELDS)}")
        return all(scores[field] <= bound for field, bound in bounds.items())

    def _activity(self, audio: Audio) -> np.ndarray:
        """
        Binary speaker activity per window, as (windows, frames, speakers).
        """

        if self._model is None:
            self._model = Model.from_pretrained(CHECKPOINT).to(self.device).eval()
            specifications = self._model.specifications
            self._powerset = Powerset(
                len(specifications.classes), specifications.powerset_max_classes
            ).to(self.device)

        y = np.asarray(audio["audio"], dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        # resampled on the model's device: the Kaiser kernel is a long convolution, and on one CPU
        # thread it cost more than the model itself
        waveform = torch.from_numpy(y)[None].to(self.device)
        if audio["sample_rate"] != SAMPLE_RATE:
            # Kaiser-best rather than the default kernel: the aliasing the cheap one leaves
            # near Nyquist reads as extra speaker activity, inflating both scores by ~30%.
            waveform = torchaudio.functional.resample(
                waveform, audio["sample_rate"], SAMPLE_RATE, **RESAMPLE)

        window = int(self._model.specifications.duration * SAMPLE_RATE)
        y = waveform[0]
        if len(y) <= window:
            chunks = [torch.nn.functional.pad(y, (0, window - len(y)))]
        else:
            starts = range(0, len(y) - window + 1, int(self.step * SAMPLE_RATE))
            chunks = [y[start:start + window] for start in starts] + [y[-window:]]

        with torch.inference_mode():
            batch = torch.stack(chunks)[:, None]
            activity = self._powerset.to_multilabel(self._model(batch))
        return activity.cpu().numpy()

