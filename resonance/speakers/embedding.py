"""Speaker embeddings of a few clips per source."""

from pathlib import Path
from typing import Callable

import numpy as np

EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAMPLE_RATE = 16000
CLIP_SECONDS = 8.0              # the model saturates well before this


def longest(paths: list[Path], count: int) -> list[Path]:
    """The longest clips of one source, by file size: a proxy that needs no decode."""

    return sorted(paths, key=lambda path: -path.stat().st_size)[:count]


def embed(sources: list[tuple[str, list[Path]]], advance: Callable[[int], None]) -> dict[str, np.ndarray]:
    """The per-clip vectors of each source, L2-normalised, as {source: (clips, dimensions)}.

    Kept per clip rather than averaged: a centroid blurs a source holding more than one voice. Runs in a process
    pinned to one GPU (resonance/pipeline/gpu.py), so the device is whatever that process sees.
    """

    import librosa
    import torch
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    from resonance.metrics.metric import default_device

    model = PretrainedSpeakerEmbedding(EMBEDDING, device=torch.device(default_device()))
    embeddings = {}
    for source, paths in sources:
        vectors = []
        for path in paths:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
            with torch.no_grad():
                vector = model(torch.tensor(audio, dtype=torch.float32)[None, None])[0]
            vectors.append(vector / np.linalg.norm(vector))
        if vectors:
            embeddings[source] = np.stack(vectors)
        advance(1)
    return embeddings
