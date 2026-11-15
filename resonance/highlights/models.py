"""The two model scores of a clip: how aroused the speaker sounds, and whose voice it is."""

from typing import Callable, Iterator

import numpy as np

from resonance.io.audio import KAISER_BEST, read_audio

AROUSAL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SPEAKER = "microsoft/wavlm-base-plus-sv"
RATE = 16000
BATCH_CLIPS, BATCH_SECONDS = 16, 100.0      # most per forward pass; 7 GB of GPU memory


def score_clips(paths: list[str], advance: Callable[[int], None]) -> tuple[np.ndarray, np.ndarray]:
    """
    (arousal, embeddings) for the clips, in order: arousal as (clips,), L2-normalised speaker embeddings as
    (clips, dimensions). One process on one GPU.

    Neighbouring clips share a padded batch, so pass them sorted by length, longest first: 2.4 times the speed
    of one clip at a time, with the same scores.
    """

    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Model, WavLMForXVector

    from resonance.metrics.metric import default_device

    device = default_device()
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    encoder = Wav2Vec2Model.from_pretrained(AROUSAL).to(device).eval()
    # the checkpoint's regression head (arousal, dominance, valence) is not part of Wav2Vec2Model
    weights = load_file(hf_hub_download(AROUSAL, "model.safetensors"))
    size = encoder.config.hidden_size
    dense, head = torch.nn.Linear(size, size).to(device), torch.nn.Linear(size, 3).to(device)
    dense.load_state_dict({"weight": weights["classifier.dense.weight"], "bias": weights["classifier.dense.bias"]})
    head.load_state_dict({"weight": weights["classifier.out_proj.weight"], "bias": weights["classifier.out_proj.bias"]})
    speaker = WavLMForXVector.from_pretrained(SPEAKER).to(device).eval()

    arousal, embeddings = [], []
    for waves, rate in _batches(paths):
        lengths = torch.tensor([-(-len(wave) * RATE // rate) for wave in waves], device=device)
        x = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(wave) for wave in waves], batch_first=True)
        x = torchaudio.functional.resample(x.to(device), rate, RATE, **KAISER_BEST)
        real = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        x = x * real
        # the arousal model takes each clip at zero mean and unit variance over its own samples
        centred = (x - (x.sum(1) / lengths)[:, None]) * real
        spread = ((centred ** 2).sum(1) / (lengths - 1)).sqrt()
        hidden = encoder(centred / (spread[:, None] + 1e-7), attention_mask=real).last_hidden_state
        frames = encoder._get_feat_extract_output_lengths(lengths)
        valid = torch.arange(hidden.shape[1], device=device)[None] < frames[:, None]
        pooled = (hidden * valid[..., None]).sum(1) / frames[:, None]
        arousal += head(torch.tanh(dense(pooled)))[:, 0].tolist()
        vectors = speaker(input_values=x, attention_mask=real.long()).embeddings
        embeddings.append(torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
        advance(len(waves))

    return np.array(arousal), np.concatenate(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)


def _batches(paths: list[str]) -> Iterator[tuple[list[np.ndarray], int]]:
    """(waveforms, sample rate) of neighbouring clips, each group within the batch limits once padded."""

    group, rate, longest = [], 0, 0
    for path in paths:
        audio = read_audio(path)
        y = audio["audio"]
        full = len(group) == BATCH_CLIPS or max(longest, len(y)) * (len(group) + 1) > BATCH_SECONDS * rate
        if group and (full or audio["sample_rate"] != rate):
            yield group, rate
            group, longest = [], 0
        group.append(y)
        rate, longest = audio["sample_rate"], max(longest, len(y))
    if group:
        yield group, rate
