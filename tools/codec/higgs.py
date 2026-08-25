import numpy as np
import torch
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

MODEL = "eustlb/higgs-audio-v2-tokenizer"
CODEBOOKS = 8


class HiggsCodec:
    """
    Higgs Audio v2 residual codec: 24 kHz waveform <-> 8 codebooks of discrete tokens.
    """

    def __init__(self, model=MODEL, device=None):
        # the tokenizer has output channels > 65536, which MPS cannot allocate
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("mps"):
            device = "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model)
        self.model = HiggsAudioV2TokenizerModel.from_pretrained(model, device_map=device)
        self.sampling_rate = self.feature_extractor.sampling_rate

    def encode(self, audio):
        """Audio dict -> (CODEBOOKS, T) token tensor."""

        y = np.asarray(audio["audio"], dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if audio["sample_rate"] != self.sampling_rate:
            raise ValueError(
                f"Higgs expects {self.sampling_rate} Hz, got {audio['sample_rate']}"
            )

        with torch.inference_mode():
            inputs = self.feature_extractor(
                raw_audio=y, sampling_rate=self.sampling_rate, return_tensors="pt"
            ).to(self.model.device)
            tokens = self.model.encode(inputs["input_values"]).audio_codes.squeeze(0)

        assert tokens.shape[0] == CODEBOOKS
        return tokens

    def decode(self, tokens):
        """(CODEBOOKS, T) token tensor -> waveform of shape (T,) at self.sampling_rate."""

        with torch.inference_mode():
            values = self.model.decode(tokens.to(self.model.device).unsqueeze(0))
        # audio_values is (batch, channel, T); the codec is mono
        return values.audio_values[0].reshape(-1).cpu().numpy()
