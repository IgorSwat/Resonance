import numpy as np

from resonance.metrics.banding import banded_z
from resonance.stages.select import choose
from resonance.config.schema import SelectConfig
from resonance.stages.source_pass import flagged_sources
from resonance.stages.speaker_drift import drifting

ORDER = ["bandwidth", "clipping", "hum", "flatness", "multi_speaker", "nisqa", "ctc"]


def verdicts(source: str, *values: str) -> list[dict[str, str]]:
    return [{"source_id": source, "verdict": value} for value in values]


def test_source_pass_counts_only_clips_that_reached_the_metric():
    records = (verdicts("interview", "multi_speaker", "multi_speaker", "accepted", "nisqa", "too_short", "bandwidth")
               + verdicts("clean", "accepted", "accepted", "ctc", "multi_speaker")
               + verdicts("tiny", "multi_speaker", "multi_speaker")
               + verdicts("unscored", "bandwidth", "clipping", "too_long", "error", "silence"))
    # interview: 2 of 4 scored; clean: 1 of 4; tiny: under min_clips; unscored: never reached the metric
    assert flagged_sources(records, ORDER, max_flag_rate=0.3, min_clips=3) == {"interview"}
    assert flagged_sources(records, ORDER, max_flag_rate=0.25, min_clips=2) == {"interview", "clean", "tiny"}


def test_banded_z_is_per_band():
    scores = np.array([0.2, 0.4, 0.6, 0.1, 0.5, 0.9, 0.3])
    durations = np.array([4.0, 5.0, 5.9, 7.0, 8.0, 8.9, 20.0])
    z = banded_z(scores, durations)
    assert np.allclose(z[:3], (scores[:3] - 0.4) / scores[:3].std())
    assert np.allclose(z[3:6], (scores[3:6] - 0.5) / scores[3:6].std())
    assert z[6] == 0                    # alone in its band: no spread to normalise by


def test_drifting_flags_the_low_outlier():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.45, 0.05, 50), [0.05]])
    z, flagged = drifting(scores, np.full(51, 7.0), max_z=-2.0)
    assert flagged[-1] and flagged.sum() <= 3 and z[-1] < -2


def rows(speaker: str, count: int, seconds: float = 10.0) -> list[dict[str, str]]:
    return [{"name": f"{speaker}-{index}", "speaker_id": speaker, "duration": str(seconds)} for index in range(count)]


def test_select_bounds():
    pool = rows("few", 3) + rows("some", 10) + rows("many", 100)
    chosen = choose(pool, SelectConfig(min_per_speaker=4, max_per_speaker=64))
    per_speaker = {speaker: sum(row["speaker_id"] == speaker for row in chosen) for speaker in ("few", "some", "many")}
    assert per_speaker == {"few": 0, "some": 10, "many": 64}
    assert [row["name"] for row in chosen] == [row["name"] for row in pool if row in chosen]     # original order


def test_select_minutes_cap_and_stability():
    config = SelectConfig(min_per_speaker=2, max_per_speaker=64, max_minutes_per_speaker=1.0)
    chosen = choose(rows("many", 100), config)
    assert len(chosen) == 6             # 6 x 10 s fills the minute
    again = choose(rows("other", 50) + rows("many", 100), config)
    assert [row["name"] for row in again if row["speaker_id"] == "many"] == [row["name"] for row in chosen]
    assert choose(rows("many", 100), SelectConfig(min_per_speaker=2, max_minutes_per_speaker=1.0, seed=1)) != chosen
