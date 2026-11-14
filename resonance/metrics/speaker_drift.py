from typing import override

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from resonance.io.audio import Audio
from resonance.metrics.metric import Metric, default_device

CHECKPOINT = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
FIELDS = ("similarity",)


class SpeakerDriftMetric(Metric):
    """
    Lowest voice similarity between two moments of a clip (ECAPA-TDNN, 192-d embeddings).

    The complement to MultiSpeakerMetric, which asks *who speaks when* inside a 10 s window and
    so merges two similar voices taking clean turns into one speaker slot. This asks whether the
    voice itself changes, and never compares timing: windows are embedded independently and the
    least similar pair is reported, so a second speaker is visible however the turns are laid
    out. Overlapped speech is the case it handles worst — a mixture embeds as neither voice.

    Silent windows are dropped rather than embedded: ECAPA maps near-silence to an arbitrary
    point far from any voice, which alone would drive the score for every clip with a pause.

    Read knowledge/emilia.md §3 before relying on this: windowed speaker embeddings against a
    speaker anchor were already measured there at AUC 0.655 on unbiased Emilia clips. The 2 s
    window is the longest of the four tried there and the best of them (0.801 vs 0.715 at
    0.75 s) — embedding quality degrades faster than short intrusions are recovered.
    """

    def __init__(self, device: str | None = None, window: float = 2.0, hop: float = 1.0, silence_db: float = -45.0) -> None:
        self.device = device or default_device()
        self.window = window
        self.hop = hop
        self.silence_db = silence_db
        self._model = None

    @override
    def evaluate(self, audio: Audio, transcript: str | None = None) -> dict[str, float]:
        """
        Minimum pairwise cosine similarity over the clip's windows; lower is worse.

        A clip too short or too quiet to yield two speech windows scores 1.0 — no evidence of a
        second voice is not evidence of one, and the stage must not reject on absence of data.
        """

        embeddings = self._embed(audio)
        if len(embeddings) < 2:
            return {"similarity": 1.0}
        similarity = embeddings @ embeddings.T
        return {"similarity": float(similarity[np.triu_indices(len(embeddings), 1)].min())}

    @override
    def passes(self, scores: dict[str, float], bounds: float) -> bool:
        """`bounds` is the lowest acceptable similarity. The pipeline judges by z within duration bands instead."""

        return scores["similarity"] >= bounds

    def _embed(self, audio: Audio) -> np.ndarray:
        """
        L2-normalized ECAPA embedding per speech window, as (windows, 192).
        """

        if self._model is None:
            # speechbrain derives device_type for cpu and cuda only, and raises on any other
            # device string, so anything else is loaded on the CPU and moved afterwards.
            known = self.device == "cpu" or self.device.startswith("cuda")
            self._model = EncoderClassifier.from_hparams(
                source=CHECKPOINT, run_opts={"device": self.device if known else "cpu"})
            if not known:
                self._model.device = self.device
                for module in self._model.mods.values():
                    if module is not None:
                        module.to(self.device)

        y = np.asarray(audio["audio"], dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        # resampled and windowed on the model's device, one sync for the whole clip
        waveform = torch.from_numpy(y)[None].to(self.device)
        if audio["sample_rate"] != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], SAMPLE_RATE)

        size, hop = int(self.window * SAMPLE_RATE), int(self.hop * SAMPLE_RATE)
        if waveform.shape[1] < size:
            return np.empty((0, 0))
        windows = waveform[0].unfold(0, size, hop)
        floor = 10 ** (self.silence_db / 20)
        windows = windows[windows.square().mean(dim=1).sqrt() > floor]
        if len(windows) < 2:
            return np.empty((0, 0))

        with torch.inference_mode():
            embeddings = self._model.encode_batch(windows).squeeze(1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.cpu().numpy()

