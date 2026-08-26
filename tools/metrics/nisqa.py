from typing import override

import numpy as np
import torch
# torchmetrics' public entry point pins the model to the CPU, so its parts are driven directly
# instead; see NisqaMetric._predict.
from torchmetrics.functional.audio.nisqa import (
    _get_librosa_melspec,
    _load_nisqa_model,
    _segment_specs,
)

from tools.metrics.metric import Metric

# NISQA emits its five heads in this order — discontinuity before coloration, which is not
# the order the paper lists them in.
FIELDS = ("mos", "noisiness", "discontinuity", "coloration", "loudness")


class NisqaMetric(Metric):
    """
    Single-ended speech quality prediction (NISQA v2.0), scored 1-5 on five dimensions.
    """

    def __init__(self, min_duration=0.15, max_duration=50.0, device=None):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model = None
        self._args = None

    @override
    def evaluate(self, audio, transcript=None):
        y = np.asarray(audio["audio"], dtype=np.float32)
        sample_rate = audio["sample_rate"]
        if y.ndim > 1:
            y = y.mean(axis=1)
        duration = len(y) / sample_rate
        if not self.min_duration <= duration <= self.max_duration:
            raise ValueError(
                f"NISQA needs {self.min_duration}-{self.max_duration} s of audio, got {duration:.2f} s"
            )

        return dict(zip(FIELDS, self._predict(y.reshape(1, -1), sample_rate)))

    def _predict(self, waveform, sample_rate):
        """NISQA's own forward, submodule by submodule, so the model can leave the CPU.

        torchmetrics loads the checkpoint with map_location='cpu' and never moves it, which
        leaves the framewise CNN — ~245 windows per clip, each a stack of six tiny convolutions
        — on one core, at ~15x the GPU's cost. The mel spectrogram stays on the CPU because it
        is librosa's.
        """

        if self._model is None:
            model, self._args = _load_nisqa_model()
            self._model = model.to(self.device).eval()
        melspec = _get_librosa_melspec(waveform, sample_rate, self._args)
        spec, windows = _segment_specs(torch.from_numpy(melspec), self._args)
        windows = windows.expand(spec.shape[0])

        # NISQA builds its padding masks with torch.arange, which without this lands them on
        # the CPU beside activations the model has already moved.
        with torch.inference_mode(), torch.device(self.device):
            x = self._model.cnn(spec.to(self.device), windows)      # packing needs CPU lengths
            x, counts = self._model.time_dependency(x, windows.to(self.device))
            scores = torch.cat([pool(x, counts) for pool in self._model.pool_layers], dim=1)
        return scores.reshape(-1).tolist()

    @override
    def validate(self, audio, lbound=None, rbound=None, transcript=None):
        lbound = DEFAULT_LBOUND if lbound is None else lbound
        rbound = {} if rbound is None else rbound
        unknown = (set(lbound) | set(rbound)) - set(FIELDS)
        if unknown:
            raise KeyError(f"Unknown NISQA dimensions: {sorted(unknown)}; expected {list(FIELDS)}")

        try:
            scores = self.evaluate(audio)
        except Exception as error:
            print(f"NisqaMetric failed: {error}")
            return False
        return all(scores[field] >= bound for field, bound in lbound.items()) and all(
            scores[field] <= bound for field, bound in rbound.items()
        )


# Calibrated on 600 files/corpus: both bounds sit at or below LJSpeech's observed floor
# (discontinuity min 3.48, coloration min 3.76), so the clean reference corpus is barely
# touched. Re-calibrate per corpus — the note advises percentile gates, not shared absolutes.
DEFAULT_LBOUND = {"discontinuity": 3.75, "coloration": 3.5}
