"""Resume state: a `<stage>.done` marker beside each stage's output, holding the stage's summary."""

import json
import os
from pathlib import Path


def marker(work_dir: Path, stage: str) -> Path:
    return work_dir / f"{stage}.done"


def done(work_dir: Path, stage: str) -> bool:
    return marker(work_dir, stage).exists()


def mark(work_dir: Path, stage: str, summary: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary = work_dir / f"{stage}.done.tmp"
    temporary.write_text(json.dumps(summary))
    os.replace(temporary, marker(work_dir, stage))


def clear(work_dir: Path, stage: str) -> None:
    marker(work_dir, stage).unlink(missing_ok=True)


def summary(work_dir: Path, stage: str) -> dict | None:
    path = marker(work_dir, stage)
    return json.loads(path.read_text()) if path.exists() else None
