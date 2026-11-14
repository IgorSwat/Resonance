"""What a stage may say while it runs. Workers never print: they send Events, and the Console renders them."""

import dataclasses
import time
from typing import Callable

PROGRESS_INTERVAL = 0.5     # seconds between progress events from one stage, so a fast loop can't flood the queue


@dataclasses.dataclass
class Event:
    kind: str               # stage_start, progress, stage_done, stage_failed, batch_done, batch_failed, info, warning, error
    worker: int | None = None
    batch: str | None = None
    stage: str | None = None
    message: str = ""
    total: int | None = None
    advance: int = 0
    summary: dict = dataclasses.field(default_factory=dict)
    time: float = dataclasses.field(default_factory=time.time)


class Progress:
    """Within-stage progress, sent as throttled events; the Console owns the tqdm bar."""

    def __init__(self, reporter: "Reporter", total: int) -> None:
        self.reporter, self.total = reporter, total
        self.pending, self.sent = 0, 0.0
        reporter.emit("progress", total=total)

    def advance(self, count: int = 1) -> None:
        self.pending += count
        if time.time() - self.sent >= PROGRESS_INTERVAL:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            self.reporter.emit("progress", total=self.total, advance=self.pending)
            self.pending, self.sent = 0, time.time()


class Reporter:
    """A stage's handle on the log. `sink` is a queue's put in a worker, or the Console itself in-process."""

    def __init__(self, sink: Callable[[Event], None], worker: int | None = None) -> None:
        self.sink, self.worker = sink, worker
        self.batch: str | None = None
        self.stage: str | None = None

    def emit(self, kind: str, **fields: object) -> None:
        self.sink(Event(kind, worker=self.worker, batch=self.batch, stage=self.stage, **fields))

    def progress(self, total: int) -> Progress:
        return Progress(self, total)

    def info(self, message: str) -> None:
        self.emit("info", message=message)

    def warning(self, message: str) -> None:
        self.emit("warning", message=message)

    def error(self, message: str) -> None:
        self.emit("error", message=message)
