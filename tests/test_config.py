from pathlib import Path

import pytest

from resonance.config import loader
from resonance.config.schema import Config

BASE = "run: {name: t}\nsource: {language: en, batches: [{shard: '001', first: 0, last: 2}]}\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


def test_defaults_and_overrides(tmp_path: Path) -> None:
    config = loader.load(write(tmp_path, BASE), ["quality.workers=3", "quality.nisqa.min={mos: 3.5}",
                                                  "select.max_minutes_per_speaker=30"])
    assert isinstance(config, Config)
    assert config.quality.workers == 3 and config.quality.nisqa.min == {"mos": 3.5}
    assert config.select.max_minutes_per_speaker == 30.0
    assert config.source.batches[0].shard == "001" and config.segment.target_duration == 8.0


def test_roundtrip(tmp_path: Path) -> None:
    config = loader.load(write(tmp_path, BASE))
    assert loader.load(write(tmp_path, loader.dump(config))) == config


@pytest.mark.parametrize("text, message", [
    (BASE + "quality: {worker: 3}\n", "Unknown key(s) quality.worker"),
    (BASE + "segment: {batch_size: big}\n", "segment.batch_size"),
    ("run: {name: t}\n", "Missing required key(s) source"),
    (BASE.replace("'001'", "1"), "source.batches[0].shard"),
    (BASE + "finalize: {zip: 1}\n", "finalize.zip"),
])
def test_strict(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(loader.ConfigError, match=message.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[").replace("]", r"\]")):
        loader.load(write(tmp_path, text))


def test_missing_file_and_bad_override(tmp_path: Path) -> None:
    with pytest.raises(loader.ConfigError, match="No config file"):
        loader.load(tmp_path / "absent.yaml")
    with pytest.raises(loader.ConfigError, match="not key=value"):
        loader.load(write(tmp_path, BASE), ["quality.workers"])


def test_shipped_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs/yodas2_sidon.yaml"
    config = loader.load(path)
    # the shipped file states the defaults, so removing every optional key must change nothing
    assert loader.load(path, ["run.name=x"]).quality == Config(config.run, config.source).quality
    assert config.segment == Config(config.run, config.source).segment
