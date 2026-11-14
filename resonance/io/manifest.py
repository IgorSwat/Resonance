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
NORMALIZED = "transcription_normalized"    # optional; written beside the transcription whenever the rows carry it

Row = dict[str, str]


def read(path: Path) -> list[Row]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, **DIALECT))


def columns_of(rows: Sequence[Row], base: Sequence[str]) -> list[str]:
    """`base`, with the normalized transcript placed after the transcription when the rows have one."""

    columns = list(base)
    if rows and NORMALIZED in rows[0] and NORMALIZED not in columns:
        columns.insert(columns.index("transcription") + 1, NORMALIZED)
    return columns


def write(path: Path, rows: Sequence[Row], columns: Sequence[str] | None = None) -> None:
    """
    Written to a temporary file and renamed, so a crash never leaves half a manifest. Without `columns` it writes
    the pipeline's own: COLUMNS, plus the normalized transcript when present.
    """

    columns = columns_of(rows, COLUMNS) if columns is None else columns

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", **DIALECT)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def hours(rows: Iterable[Row]) -> float:
    return sum(float(row["duration"]) for row in rows) / 3600
