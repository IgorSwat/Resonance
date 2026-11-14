"""Reading clips, and the resampling kernel shared by the segmenter and the codecs."""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

Audio = dict[str, Any]      # {"audio": mono float32 waveform, "sample_rate": int}

# librosa's kaiser_best: the default kernel leaves aliasing near Nyquist that a cheap decimation folds straight
# back into the speech band.
KAISER_BEST = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379392,
}


def read_audio(path: Path) -> Audio:
    y, rate = sf.read(path, dtype="float32")
    return {"audio": y.mean(axis=1) if y.ndim > 1 else y, "sample_rate": rate}


def resample(y: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return y
    
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(y), rate, target, **KAISER_BEST).numpy()
