"""
The clip manifest: Resonance's `|`-separated CSV, never quoted, rows ending in CRLF.

Transcripts can hold a bare `\\r` or an escaped `|`, so manifests are read and written only through this module.
Between stages a manifest carries EXTRA columns; `finalize` trims it to STANDARD.
"""

import csv
import os
from pathlib import Path
from typing import Iterable, Sequence

DIALECT = {"delimiter": "|", "quotechar": None, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
STANDARD = ("dataset", "name", "transcription", "speaker_id", "language")
EXTRA = ("duration", "audio_path", "source_id")
COLUMNS = STANDARD + EXTRA

Row = dict[str, str]


def read(path: Path) -> list[Row]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, **DIALECT))


def write(path: Path, rows: Iterable[Row], columns: Sequence[str] = COLUMNS) -> None:
    """Written to a temporary file and renamed, so a crash never leaves half a manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", **DIALECT)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def hours(rows: Iterable[Row]) -> float:
    return sum(float(row["duration"]) for row in rows) / 3600
