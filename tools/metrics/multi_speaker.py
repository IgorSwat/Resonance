from typing import override

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.audio.utils.powerset import Powerset

from tools.metrics.metric import Metric

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

    def __init__(self, device=None, step=2.5):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.step = step
        self._model = None
        self._powerset = None

    @override
    def evaluate(self, audio, transcript=None):
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
    def validate(self, audio, lbound=None, rbound=None, transcript=None):
        rbound = DEFAULT_RBOUND if rbound is None else rbound
        unknown = set(rbound) - set(FIELDS)
        if unknown:
            raise KeyError(f"Unknown speaker scores: {sorted(unknown)}; expected {list(FIELDS)}")

        try:
            scores = self.evaluate(audio)
        except Exception as error:
            print(f"MultiSpeakerMetric failed: {error}")
            return False
        return all(scores[field] <= bound for field, bound in rbound.items())

    def _activity(self, audio):
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
        waveform = torch.from_numpy(y)[None]
        if audio["sample_rate"] != SAMPLE_RATE:
            # Kaiser-best rather than the default kernel: the aliasing the cheap one leaves
            # near Nyquist reads as extra speaker activity, inflating both scores by ~30%.
            waveform = torchaudio.functional.resample(
                waveform, audio["sample_rate"], SAMPLE_RATE, **RESAMPLE)

        window = int(self._model.specifications.duration * SAMPLE_RATE)
        y = waveform[0].numpy()
        if len(y) <= window:
            chunks = [np.pad(y, (0, window - len(y)))]
        else:
            starts = range(0, len(y) - window + 1, int(self.step * SAMPLE_RATE))
            chunks = [y[start:start + window] for start in starts] + [y[-window:]]

        with torch.inference_mode():
            batch = torch.from_numpy(np.stack(chunks)).to(self.device)[:, None]
            activity = self._powerset.to_multilabel(self._model(batch))
        return activity.cpu().numpy()


# The recall-first gate: reject any clip with a second speaker at all. Calibrated by ear over
# 49 blind-labelled clips, where every confirmed second speaker scored second >= 0.14 s and no
# negative scored above 0.08 s; the score is bimodal, so 0.0 and 0.1 differ by 1.5% of a batch.
# Raise second to ~0.5 and keep overlap at 0.0 for a tighter, precision-first gate.
DEFAULT_RBOUND = {"second": 0.0, "overlap": 0.0}
