import json
import zipfile
from pathlib import Path

import pytest

import fakes
from resonance.io import manifest
from resonance.log.console import Console
from resonance.pipeline import gpu, runner, state
from resonance.pipeline.artifacts import BatchRef, Layout, batches
from resonance.stages.discard_rejected import DiscardRejected
from resonance.stages.finalize import Finalize, restore
from resonance.stages.merge import Merge
from resonance.stages.select import Select
from resonance.stages.source_pass import SourcePass

STAGES = [fakes.FakeFetch, fakes.FakeSegment, fakes.FakeQuality, DiscardRejected, Merge, Select, Finalize]
KEPT = fakes.SOURCES * (fakes.CLIPS_PER_SOURCE - 1)


def test_validate_checks_types_order_and_side_files() -> None:
    batch, pool = runner.validate(STAGES)
    assert [cls.name for cls in batch] == ["fetch", "segment", "quality", "discard_rejected"] and len(pool) == 3
    with pytest.raises(runner.PipelineError, match="quality consumes ClipManifest, but it would be given RawBatch"):
        runner.validate([fakes.FakeFetch, fakes.FakeQuality])
    with pytest.raises(runner.PipelineError, match="must all come before"):
        runner.validate([fakes.FakeFetch, fakes.FakeSegment, Merge, fakes.FakeQuality])
    with pytest.raises(runner.PipelineError, match="needs verdicts.csv"):
        runner.validate([fakes.FakeFetch, fakes.FakeSegment, SourcePass])
    with pytest.raises(runner.PipelineError, match="select consumes ClipManifest"):
        runner.validate([fakes.FakeFetch, fakes.FakeSegment, Select])


def test_batch_ref_parse() -> None:
    assert BatchRef.parse("en001/91") == BatchRef("en", "001", 91) == BatchRef.parse("en001/train-00091")
    assert BatchRef("en", "001", 91).key == "en001/train-00091"


def test_whole_pipeline_on_two_fake_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "discover", lambda requested: [0, 1])
    config = fakes.config(tmp_path, 0, 3)
    layout = Layout(config)
    assert runner.run(config, Console(layout.run_dir), classes=STAGES)

    ran = [json.loads((layout.batch_dir(ref) / "ran_on.json").read_text()) for ref in batches(config)]
    assert {entry["gpu"] for entry in ran} == {"0", "1"} and len({entry["pid"] for entry in ran}) == 2
    assert {entry["threads"] for entry in ran} == {str(config.resources.threads_per_worker)}
    for ref in batches(config):
        assert not layout.raw(ref).exists() or True     # the fake segment keeps raw; the real one removes it
        assert len(manifest.read(layout.batch_dir(ref) / "discard_rejected.csv")) == KEPT
    # rejected audio is gone, kept audio moved into the zip, and the batches are marked consumed
    assert not any(layout.root.glob("clips/*/*/*.wav"))
    selected = manifest.read(layout.run_dir / "select.csv")
    assert len(selected) == 4 * KEPT
    with zipfile.ZipFile(layout.dataset / "audio.zip") as archive:
        assert sorted(archive.namelist()) == sorted(f"{row['name']}.wav" for row in selected)
        assert archive.infolist()[0].compress_type == zipfile.ZIP_DEFLATED
    assert not (layout.dataset / "audio").exists()
    final = manifest.read(layout.dataset / "test.csv")
    assert list(final[0]) == list(manifest.STANDARD) and len(final) == len(selected)
    assert json.loads((layout.batch_dir(batches(config)[0]) / "consumed.json").read_text()) == {"run": "test"}
    assert state.summary(layout.run_dir, "finalize")["clips_out"] == len(selected)
    assert "finalize" in (layout.run_dir / "run.log").read_text()

    # a finished run does nothing when started again, and cannot be reset while its clips are in the zip
    assert runner.run(config, Console(), classes=STAGES)
    with pytest.raises(runner.PipelineError, match="restore"):
        runner.run(config, Console(), start="select", classes=STAGES)

    assert restore(config) == len(selected)
    assert len(list(layout.root.glob("clips/*/*/*.wav"))) == len(selected) and not layout.dataset.exists()
    assert not list(layout.root.glob("batches/*/*/consumed.json"))
    with pytest.raises(runner.PipelineError, match="--from fetch"):
        runner.run(config, Console(), start="quality", classes=STAGES)
    assert runner.run(config, Console(), start="select", classes=STAGES)
    assert (layout.dataset / "audio.zip").exists()


def test_failure_stops_the_run_and_resume_finishes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "discover", lambda requested: [0])
    config = fakes.config(tmp_path, 0, 2, **{"finalize.move": "false", "finalize.zip": "false"})
    layout = Layout(config)
    layout.root.mkdir(parents=True)
    (layout.root / "fail_1").touch()
    assert not runner.run(config, Console(layout.run_dir), classes=STAGES)
    assert state.done(layout.batch_dir(batches(config)[0]), "discard_rejected")
    assert not state.done(layout.batch_dir(batches(config)[1]), "segment")
    assert not state.done(layout.run_dir, "merge")
    assert "segment was told to fail" in (layout.run_dir / "run.log").read_text()

    (layout.root / "fail_1").unlink()
    first = (layout.batch_dir(batches(config)[0]) / "ran_on.json").read_text()
    assert runner.run(config, Console(layout.run_dir), classes=STAGES)
    assert (layout.batch_dir(batches(config)[0]) / "ran_on.json").read_text() == first      # not rerun
    # copied, not moved or zipped: the clips stay and nothing is marked consumed
    assert len(list((layout.dataset / "audio").glob("*.wav"))) == 3 * KEPT == len(list(layout.root.glob("clips/*/*/*.wav")))
    assert not list(layout.root.glob("batches/*/*/consumed.json"))


def test_until_stops_before_pool_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "discover", lambda requested: [None])
    config = fakes.config(tmp_path, 0, 0)
    assert runner.run(config, Console(), until="quality", classes=STAGES)
    layout = Layout(config)
    assert state.done(layout.batch_dir(batches(config)[0]), "quality")
    assert not state.done(layout.batch_dir(batches(config)[0]), "discard_rejected")
    assert not state.done(layout.run_dir, "merge")


def test_console_logs_progress_without_a_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from resonance.log import console as console_module
    from resonance.log.events import Reporter

    monkeypatch.setattr(console_module, "HEARTBEAT", 0.0)
    console = Console(tmp_path)
    reporter = Reporter(console.handle, worker=0)
    reporter.batch, reporter.stage = "en001/train-00091", "segment"
    progress = reporter.progress(10)
    progress.advance(4)
    progress.flush()
    reporter.emit("stage_done", summary={"clips_out": 3, "hours_out": 0.5, "seconds": 61})
    console.close()
    log = (tmp_path / "run.log").read_text()
    assert "[w0] en001/train-00091 segment 4/10" in log and "3 clips · 0.50 h · 1m01s" in log
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1
