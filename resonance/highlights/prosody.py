"""How lively a clip's delivery is, from its pitch and loudness alone."""

import numpy as np

from resonance.io.audio import read_audio, resample

RATE = 16000
FIELDS = ("words_s", "voiced", "f0_median", "f0_sd", "f0_range", "loud_range")


def features(task: tuple[str, int]) -> dict[str, float]:
    """
    One clip's prosody, from (path, word count): speaking rate, voiced share, and the spread of pitch (in semitones
    around the clip's median) and of loudness. The pitch fields are NaN when too little of the clip is voiced.
    """

    import librosa

    path, words = task
    audio = read_audio(path)
    y = resample(audio["audio"], audio["sample_rate"], RATE)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=RATE, frame_length=1024)
    f0 = f0[voiced & np.isfinite(f0)]
    decibels = 20 * np.log10(librosa.feature.rms(y=y, frame_length=800, hop_length=400)[0] + 1e-6)
    loud = decibels[decibels > decibels.max() - 40]
    row = {"words_s": words / (len(y) / RATE), "voiced": float(np.mean(voiced))}

    if len(f0) < 20:
        return {**row, "f0_median": np.nan, "f0_sd": np.nan, "f0_range": np.nan, "loud_range": np.nan}
    
    semitones = 12 * np.log2(f0 / np.median(f0))

    return {**row, "f0_median": float(np.median(f0)), "f0_sd": float(semitones.std()),
            "f0_range": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
            "loud_range": float(np.percentile(loud, 95) - np.percentile(loud, 10))}
