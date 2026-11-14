"""Reading a recording and finding its speech, in worker processes ahead of Whisper."""

import time
from pathlib import Path

from resonance.datasets.yodas2_sidon import read_recording
from resonance.io.audio import resample

MODEL_RATE = 16000              # what Whisper and Silero both assume: mono float32 at 16 kHz


def setup(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def prepare(task: tuple[Path, float]) -> dict:
    """
    Read, resample and VAD one recording: {path, loaded, y, speech, spent} or {path, error, spent}.

    A process rather than a thread: faster-whisper's VAD steps through the audio in Python and holds the GIL, so
    threads queue behind each other. `loaded` is None for a recording without captions.
    """

    from faster_whisper.vad import VadOptions, get_speech_timestamps

    meta_path, margin = task
    clock = time.perf_counter()
    try:
        loaded = read_recording(meta_path, margin)
    except Exception as error:
        return {"path": meta_path, "error": str(error), "spent": {"read": time.perf_counter() - clock}}
    spent = {"read": time.perf_counter() - clock}
    if loaded is None:
        return {"path": meta_path, "loaded": None, "spent": spent}
    _, audio, rate, _, _ = loaded
    clock = time.perf_counter()
    y = resample(audio, rate, MODEL_RATE)
    spent["resample"] = time.perf_counter() - clock
    clock = time.perf_counter()
    options = VadOptions(min_silence_duration_ms=100, speech_pad_ms=0)
    speech = [(s["start"] / MODEL_RATE, s["end"] / MODEL_RATE) for s in get_speech_timestamps(y, options)]
    spent["vad"] = time.perf_counter() - clock
    return {"path": meta_path, "loaded": loaded, "y": y, "speech": speech, "spent": spent}
