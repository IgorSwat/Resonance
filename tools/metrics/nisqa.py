from typing import override

import librosa
import numpy as np
import torch
# torchmetrics' public entry point pins the model to the CPU, so its parts are driven directly
# instead; see NisqaMetric._predict.
from torchmetrics.functional.audio.nisqa import _load_nisqa_model

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
        self._mel = None                # (sample rate, filterbank, window), built on first use

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
        — on one core, at ~15x the GPU's cost.
        """

        if self._model is None:
            model, self._args = _load_nisqa_model()
            self._model = model.to(self.device).eval()
        spec, windows = self._segments(self._melspec(waveform, sample_rate))
        windows = windows.expand(spec.shape[0])

        # NISQA builds its padding masks with torch.arange, which without this lands them on
        # the CPU beside activations the model has already moved.
        with torch.inference_mode(), torch.device(self.device):
            x = self._model.cnn(spec.to(self.device), windows)      # packing needs CPU lengths
            x, counts = self._model.time_dependency(x, windows.to(self.device))
            scores = torch.cat([pool(x, counts) for pool in self._model.pool_layers], dim=1)
        return scores.reshape(-1).tolist()

    def _melspec(self, waveform, sample_rate):
        """torchmetrics' _get_librosa_melspec, on the model's device.

        librosa's STFT and filterbank product ran on one CPU core at ~70 ms a clip, most of the
        stage's cost; this is the same transform — periodic Hann window centred in n_fft,
        reflect padding, magnitude, librosa's own Slaney filterbank, dB with an 80 dB floor below
        each clip's peak — at a fraction of it.
        """

        args = self._args
        hop, win, n_fft = (int(sample_rate * args["ms_hop_length"]),
                           int(sample_rate * args["ms_win_length"]), args["ms_n_fft"])
        if self._mel is None or self._mel[0] != sample_rate:
            basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=args["ms_n_mels"],
                                        fmin=0.0, fmax=args["ms_fmax"], htk=False, norm="slaney")
            self._mel = (sample_rate, torch.from_numpy(basis).to(self.device),
                         torch.hann_window(win, device=self.device))
            device = torch.device(self.device)
            if device.type == "cuda":
                # every clip length is a new cuFFT plan, cached with its GPU workspace up to 4096
                # of them; with several workers sharing the card that runs it out of memory
                index = torch.cuda.current_device() if device.index is None else device.index
                torch.backends.cuda.cufft_plan_cache[index].max_size = 16
        _, basis, window = self._mel

        y = torch.from_numpy(np.ascontiguousarray(waveform)).to(self.device)
        try:
            spectrum = torch.stft(y, n_fft, hop, win, window, center=True, pad_mode="reflect",
                                  return_complex=True)
        except RuntimeError:
            # cuFFT still fails to allocate a plan now and then on a full card, and a raise here
            # rejects the clip; the CPU computes the same transform
            spectrum = torch.stft(y.cpu(), n_fft, hop, win, window.cpu(), center=True,
                                  pad_mode="reflect", return_complex=True).to(self.device)
        magnitude = spectrum.abs()
        db = 20 * torch.log10((basis @ magnitude).clamp(min=1e-4))
        return torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)

    def _segments(self, melspec):
        """torchmetrics' _segment_specs on the model's device: overlapping windows, zero-padded."""

        length, hop, most = (self._args["ms_seg_length"], self._args["ms_seg_hop_length"],
                             self._args["ms_max_segments"])
        if melspec.shape[2] < length:
            raise RuntimeError("Input signal is too short.")
        x = melspec.unfold(2, length, hop).transpose(1, 2)    # (batch, windows, mels, length)
        count = x.shape[1]
        if most < count:
            raise RuntimeError("Maximum number of mel spectrogram windows exceeded. Use shorter audio.")
        return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, most - count)), torch.tensor(count)

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
