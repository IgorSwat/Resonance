import json
import shutil
import sys
import time
from pathlib import Path

import pytest
import yaml

from resonance.config.schema import UploadConfig
from resonance.datasets import yodas2_sidon
from resonance.orchestrate import drive, supervisor
from resonance.orchestrate.chunks import Chunk, plan
from resonance.pipeline.artifacts import BatchRef

SHARDS = {"en000": 132, "en001": 50, "en004": 132}


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yodas2_sidon, "shards", lambda language: sorted(SHARDS))
    monkeypatch.setattr(yodas2_sidon, "available",
                        lambda shard: [f"train-{index:05d}.tar.gz" for index in range(SHARDS[shard])])


def test_chunk_names_and_ranges() -> None:
    chunk = Chunk("en", "004", 100, 131, skipped=[101, 102, 131])
    assert chunk.name == "en004_132"
    assert [(span.first, span.last) for span in chunk.ranges()] == [(100, 100), (103, 130)]


def test_plan_follows_the_drive_names(corpus: None) -> None:
    assert [chunk.name for chunk in plan("en", 50)] == [
        "en000_50", "en000_100", "en000_132", "en001_50", "en004_50", "en004_100", "en004_132"]
    assert [chunk.name for chunk in plan("en", 50, shards=["001"])] == ["en001_50"]


def test_plan_starts_at_a_batch(corpus: None) -> None:
    chunks = plan("en", 50, start=BatchRef("en", "004", 100))
    assert [(chunk.name, chunk.first, chunk.last) for chunk in chunks] == [("en004_132", 100, 131)]
    first = plan("en", 50, start=BatchRef("en", "004", 70))[0]
    assert (first.name, first.first, first.last) == ("en004_100", 70, 99)
    with pytest.raises(ValueError, match="not a en batch"):
        plan("en", 50, start=BatchRef("de", "000", 0))


def test_progress_line() -> None:
    line = drive.progress_line({"bytes": 13_000_000_000, "totalBytes": 52_000_000_000, "speed": 85e6, "eta": 460})
    assert line == "13.0/52.0 GB (25%), 85 MB/s, ETA 7m40s"
    assert drive.progress_line({"bytes": 0, "totalBytes": 0}) == ""


def test_read_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("old line\n")
    offset = path.stat().st_size
    events = [{"kind": "stage_start", "batch": "en004/train-00100", "stage": "segment"},
              {"kind": "stage_start", "batch": "en004/train-00101", "stage": "segment"},
              {"kind": "stage_failed", "batch": "en004/train-00101", "stage": "segment"},
              {"kind": "batch_failed", "batch": "en004/train-00101"},
              {"kind": "stage_failed", "batch": None, "stage": "select"}]
    with open(path, "a") as handle:
        handle.writelines(json.dumps(event) + "\n" for event in events)
    assert supervisor.read_events(path, offset) == (["en004/train-00100"], ["en004/train-00101"], True)


needs_rclone = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")


@needs_rclone
def test_upload_checks_and_lists(tmp_path: Path) -> None:
    config = UploadConfig(path=str(tmp_path / "drive" / "train/{language_name}/yodas2_sidon/{run}"), stats_seconds=1)
    dataset, highlights = tmp_path / "dataset", tmp_path / "highlights"
    for folder, names in ((dataset, ["audio.zip", "en004_132.csv"]), (highlights, ["audio.zip", "scores.csv"])):
        folder.mkdir()
        for name in names:
            (folder / name).write_bytes(name.encode() * 1000)
    lines: list[str] = []
    assert drive.uploaded(config, "en", ["en004_132"]) == set()
    drive.upload(config, "en", "en004_132", dataset, highlights, lines.append)
    remote = tmp_path / "drive/train/english/yodas2_sidon/en004_132"
    assert (remote / "en004_132.csv").exists() and (remote / "highlights" / "scores.csv").exists()
    assert drive.uploaded(config, "en", ["en004_132", "en004_100"]) == {"en004_132"}
    assert any("verified" in line for line in lines)


CHILD = """
import json, sys, time
from pathlib import Path
run_dir, mode = Path(sys.argv[1]), sys.argv[2]
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run.log").write_text("started\\n")
if mode == "hang":
    time.sleep(600)
if mode == "fail":
    with open(run_dir / "events.jsonl", "a") as handle:
        handle.write(json.dumps({"kind": "stage_start", "batch": "en004/train-00101", "stage": "segment"}) + "\\n")
        handle.write(json.dumps({"kind": "batch_failed", "batch": "en004/train-00101"}) + "\\n")
    sys.exit(1)
(run_dir / "dataset").mkdir(exist_ok=True)
(run_dir / "dataset" / "audio.zip").write_bytes(b"zip" * 100)
(run_dir / "dataset" / "en004_132.csv").write_text("name|transcription\\n")
(run_dir / "highlights").mkdir(exist_ok=True)
(run_dir / "highlights" / "scores.csv").write_text("x\\n")
(run_dir / "finalize.done").write_text(json.dumps({"clips_out": 10, "hours_out": 1.5, "audio_gb": 0.1}))
"""


def orchestrator(tmp_path: Path, behaviour, **settings: object) -> supervisor.Orchestrator:
    """An orchestrator over a fake corpus whose pipeline is CHILD, run in the mode `behaviour(chunk)` picks."""

    config = tmp_path / "c.yaml"
    upload = {"path": f"{tmp_path / 'drive'}/train/{{language_name}}/yodas2_sidon/{{run}}", "stats_seconds": 1}
    config.write_text(yaml.safe_dump({
        "run": {"name": "base", "data_root": str(tmp_path / "data")},
        "source": {"language": "en", "batches": [{"shard": "004", "first": 0, "last": 0}]},
        "orchestrate": {"poll_seconds": 0.1, "stall_minutes": 0.05, "min_free_gb": 0, "upload": upload, **settings}}))
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    clips = tmp_path / "data/clips/en004/train-00100"
    clips.mkdir(parents=True)

    def command(chunk: Chunk, preflight: bool) -> list[str]:
        return [sys.executable, str(script), str(tmp_path / "data/runs" / chunk.name), behaviour(chunk)]

    return supervisor.Orchestrator(config, [], "en", BatchRef("en", "004", 100), command=command)


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch, corpus: None) -> None:
    monkeypatch.setattr(supervisor, "BACKOFF", (0, 0, 0))


@needs_rclone
def test_a_failing_batch_is_quarantined_and_the_chunk_delivered(tmp_path: Path, fast: None) -> None:
    runner = orchestrator(tmp_path, lambda chunk: "ok" if 101 in chunk.skipped else "fail")
    runner.run()
    entry = json.loads((tmp_path / "data/orchestrator/en.json").read_text())["chunks"]["en004_132"]
    assert entry["state"] == "done" and entry["skipped"] == [101] and entry["restarts"] == 3
    assert entry["failures"] == {"en004/train-00101": 3} and entry["hours"] == 1.5
    remote = tmp_path / "drive/train/english/yodas2_sidon/en004_132"
    assert (remote / "audio.zip").exists() and (remote / "highlights/scores.csv").exists()
    assert not (tmp_path / "data/runs/en004_132/dataset").exists()
    assert not (tmp_path / "data/clips/en004/train-00100").exists()
    # a second start finds the chunk on Drive and has nothing to do
    assert supervisor.Orchestrator(tmp_path / "c.yaml", [], "en", BatchRef("en", "004", 100)).todo() == []


@needs_rclone
def test_a_hang_is_killed_and_restarted(tmp_path: Path, fast: None) -> None:
    attempts: list[str] = []

    def behaviour(chunk: Chunk) -> str:
        attempts.append("hang" if not attempts else "ok")
        return attempts[-1]

    started = time.time()
    orchestrator(tmp_path, behaviour).run()
    assert attempts == ["hang", "ok"] and time.time() - started < 60
    entry = json.loads((tmp_path / "data/orchestrator/en.json").read_text())["chunks"]["en004_132"]
    assert entry["state"] == "done" and entry["other_failures"] == 1


def test_repeated_failures_without_a_batch_stop(tmp_path: Path, fast: None) -> None:
    runner = orchestrator(tmp_path, lambda chunk: "crash")
    script = tmp_path / "child.py"
    script.write_text("import sys; sys.exit(3)")
    with pytest.raises(supervisor.Stop, match="ended \\(3\\) 3 times"):
        runner.run()
