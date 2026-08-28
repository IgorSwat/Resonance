from typing import override

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from tools.metrics.metric import Metric

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

    def __init__(self, device=None, window=2.0, hop=1.0, silence_db=-45.0):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.window = window
        self.hop = hop
        self.silence_db = silence_db
        self._model = None

    @override
    def evaluate(self, audio, transcript=None):
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
    def validate(self, audio, lbound=None, rbound=None, transcript=None):
        lbound = DEFAULT_LBOUND if lbound is None else lbound
        unknown = set(lbound) - set(FIELDS)
        if unknown:
            raise KeyError(f"Unknown drift scores: {sorted(unknown)}; expected {list(FIELDS)}")

        try:
            scores = self.evaluate(audio)
        except Exception as error:
            print(f"SpeakerDriftMetric failed: {error}")
            return False
        return all(scores[field] >= bound for field, bound in lbound.items())

    def _embed(self, audio):
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
        waveform = torch.from_numpy(y)[None]
        if audio["sample_rate"] != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], SAMPLE_RATE)

        y = waveform[0]
        size, hop = int(self.window * SAMPLE_RATE), int(self.hop * SAMPLE_RATE)
        windows = [y[start:start + size] for start in range(0, len(y) - size + 1, hop)]
        floor = 10 ** (self.silence_db / 20)
        windows = [w for w in windows if float(w.square().mean().sqrt()) > floor]
        if len(windows) < 2:
            return np.empty((0, 0))

        with torch.inference_mode():
            batch = torch.stack(windows).to(self.device)
            embeddings = self._model.encode_batch(batch).squeeze(1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.cpu().numpy()


# Measured on 100 random Emilia EN-B000007 clips that already passed MultiSpeakerMetric: p50
# 0.423, p10 0.206, p5 0.157, p1 0.051. The score is unimodal there, so unlike
# MultiSpeakerMetric's bimodal `second` no bound separates two populations, and 0.16 is only the
# p5 outlier cut. Re-derive per corpus, and treat a rejection as a clip to listen to.
DEFAULT_LBOUND = {"similarity": 0.16}
