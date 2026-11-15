"""The highlight lists, from per-clip scores. Pure NumPy: no audio, no model."""

import numpy as np

MIN_VOICED = 0.3                # guard rails of the expressive score: a clip outside them is mostly music,
MAX_F0_SD = 6.0                 # noise or a pitch-tracking failure, and its spread is not expressiveness
WORDS_PER_SECOND = (1.5, 5.5)
NEIGHBOURS = 5                  # a voice is unique when even its closest voices are far


def expressive(scores: dict[str, np.ndarray]) -> np.ndarray:
    """
    How lively each clip sounds: the z-scores of pitch spread, pitch range, loudness range and arousal, summed.
    NaN for a clip outside the guard rails or without pitch.
    """

    def z(values: np.ndarray) -> np.ndarray:
        return (values - np.nanmean(values)) / np.nanstd(values)

    total = z(scores["f0_sd"]) + z(scores["f0_range"]) + z(scores["loud_range"]) + z(scores["arousal"])
    sane = ((scores["voiced"] >= MIN_VOICED) & (scores["f0_sd"] <= MAX_F0_SD)
            & (scores["words_s"] >= WORDS_PER_SECOND[0]) & (scores["words_s"] <= WORDS_PER_SECOND[1]))
    return np.where(sane, total, np.nan)


def top(values: np.ndarray, speakers: list[str], count: int, per_speaker: int) -> list[int]:
    """Indices of the `count` highest values, skipping NaN and taking at most `per_speaker` clips of one speaker."""

    chosen, taken = [], {}
    for index in np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable"):
        if len(chosen) == count or np.isnan(values[index]):
            break
        if taken.get(speakers[index], 0) < per_speaker:
            chosen.append(int(index))
            taken[speakers[index]] = taken.get(speakers[index], 0) + 1
    return chosen


def voice_unique(embeddings: np.ndarray, speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    (unique, typical), both per clip. `unique` is the clip's speaker's distance from the pool: 1 minus the mean
    cosine between the speaker's centroid and its NEIGHBOURS closest other centroids. `typical` is the cosine
    between the clip and its own speaker's centroid, so the most typical clip of a speaker can stand for it.
    """

    names = sorted(set(speakers))
    position = {name: index for index, name in enumerate(names)}
    owner = np.array([position[speaker] for speaker in speakers])
    centroids = np.zeros((len(names), embeddings.shape[1]), dtype=np.float32)
    np.add.at(centroids, owner, embeddings)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.sort(similarity, axis=1)[:, -min(NEIGHBOURS, len(names) - 1):] if len(names) > 1 else np.zeros((1, 1))
    unique = 1 - nearest.mean(axis=1)
    return unique[owner], np.einsum("ij,ij->i", embeddings, centroids[owner])


def unique_voices(unique: np.ndarray, typical: np.ndarray, speakers: list[str], count: int) -> list[int]:
    """Indices of one clip each for the `count` most unique speakers: every speaker's most typical clip."""

    best: dict[str, int] = {}
    for index, speaker in enumerate(speakers):
        if speaker not in best or typical[index] > typical[best[speaker]]:
            best[speaker] = index
    return sorted(best.values(), key=lambda index: -unique[index])[:count]
