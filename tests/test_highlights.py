import numpy as np

from resonance.highlights import ranking


def test_expressive_sums_z_scores_inside_the_guard_rails() -> None:
    scores = {"f0_sd": np.array([1.0, 2.0, 3.0, 9.0, 2.0, np.nan]), "f0_range": np.array([3.0, 6.0, 9.0, 20.0, 6.0, np.nan]),
              "loud_range": np.array([10.0, 12.0, 14.0, 12.0, 12.0, 12.0]), "arousal": np.array([0.2, 0.4, 0.6, 0.4, 0.4, 0.9]),
              "voiced": np.array([0.6, 0.6, 0.6, 0.6, 0.1, 0.6]), "words_s": np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])}
    result = ranking.expressive(scores)
    assert result[2] > result[1] > result[0]
    assert np.isnan(result[3]) and np.isnan(result[4]) and np.isnan(result[5])   # wild pitch, unvoiced, no pitch


def test_top_caps_clips_per_speaker_and_skips_nan() -> None:
    values = np.array([9.0, 8.0, 7.0, np.nan, 6.0, 5.0])
    speakers = ["a", "a", "a", "b", "b", "c"]
    assert ranking.top(values, speakers, count=4, per_speaker=2) == [0, 1, 4, 5]
    assert ranking.top(values, speakers, count=2, per_speaker=1) == [0, 4]
    assert ranking.top(np.array([np.nan, np.nan]), ["a", "b"], 5, 2) == []


def test_voice_unique_finds_the_outlier_speaker_and_its_typical_clip() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=16)
    voices = {name: common + 0.3 * rng.normal(size=16) for name in "abcdefg"}
    voices["odd"] = -common + rng.normal(size=16)
    speakers, vectors = [], []
    for name, voice in voices.items():
        for clip in range(4):
            vector = voice + (2.0 if clip == 3 else 0.1) * rng.normal(size=16)      # the last clip is atypical
            speakers.append(name)
            vectors.append(vector / np.linalg.norm(vector))
    unique, typical = ranking.voice_unique(np.array(vectors, dtype=np.float32), speakers)
    picked = ranking.unique_voices(unique, typical, speakers, count=3)
    assert speakers[picked[0]] == "odd" and len({speakers[index] for index in picked}) == 3
    assert all(index % 4 != 3 for index in picked)
    assert len(set(np.round(unique[:4], 6))) == 1                                   # one value per speaker
