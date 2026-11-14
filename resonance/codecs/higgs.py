"""Higgs Audio v2 residual codec: 24 kHz waveform -> 8 codebooks of discrete tokens."""

import numpy as np

from resonance.codecs.codec import Codec
from resonance.io.audio import Audio, resample

MODEL = "eustlb/higgs-audio-v2-tokenizer"
CODEBOOKS = 8


class HiggsCodec(Codec):
    """
    HiggsAudio tokernizer v2: 25Hz, 8 codebooks, vocab size 1024 (stored as int16).

    Only CUDA reproduces existing tokens; the tokenizer has output channels > 65536, which MPS cannot allocate,
    so anything but CUDA runs on the CPU.
    """

    name = "higgs"

    def __init__(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = HiggsAudioV2TokenizerModel.from_pretrained(MODEL, device_map=device)
        self.sample_rate = self.feature_extractor.sampling_rate

    def encode(self, audio: Audio) -> np.ndarray:
        import torch

        y = resample(np.asarray(audio["audio"], dtype=np.float32), audio["sample_rate"], self.sample_rate)
        with torch.inference_mode():
            inputs = self.feature_extractor(raw_audio=y, sampling_rate=self.sample_rate,
                                            return_tensors="pt").to(self.model.device)
            tokens = self.model.encode(inputs["input_values"]).audio_codes.squeeze(0)
        assert tokens.shape[0] == CODEBOOKS
        return tokens.cpu().numpy().astype(np.int16)
