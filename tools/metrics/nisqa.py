from typing import override

import numpy as np
import torch
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment

from tools.metrics.metric import Metric

# NISQA emits its five heads in this order — discontinuity before coloration, which is not
# the order the paper lists them in.
FIELDS = ("mos", "noisiness", "discontinuity", "coloration", "loudness")


class NisqaMetric(Metric):
    """
    Single-ended speech quality prediction (NISQA v2.0), scored 1-5 on five dimensions.
    """

    def __init__(self, min_duration=0.15, max_duration=50.0):
        self.min_duration = min_duration
        self.max_duration = max_duration

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

        scores = non_intrusive_speech_quality_assessment(torch.from_numpy(y), sample_rate)
        return dict(zip(FIELDS, scores.tolist()))

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
