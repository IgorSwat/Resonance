import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

MODEL = "eustlb/higgs-audio-v2-tokenizer"
CODEBOOKS = 8

# librosa's kaiser_best, as in tools/metrics/multi_speaker.py — the default kernel leaves
# aliasing near Nyquist that a cheap decimation folds straight back into the speech band
RESAMPLE = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379392,
}


def to_codec_rate(audio, rate):
    """
    Resample an audio dict to `rate`, so a source at another sample rate can be encoded.

    Downsampling drops everything above the new Nyquist, which the codec could not represent
    at `rate` anyway; passing the samples through unconverted would instead reinterpret the
    timebase and stretch the clip. Emilia is 24 kHz throughout the batches measured so far, so
    anything this converts is worth tracing upstream rather than accepting silently.
    """

    if audio["sample_rate"] == rate:
        return audio
    y = np.asarray(audio["audio"], dtype=np.float32)
    waveform = torch.from_numpy(y.T if y.ndim > 1 else y)
    resampled = torchaudio.functional.resample(
        waveform, audio["sample_rate"], rate, **RESAMPLE
    ).numpy()
    return audio | {"audio": resampled.T if y.ndim > 1 else resampled, "sample_rate": rate}


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
