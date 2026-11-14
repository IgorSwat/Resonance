"""Renders Events: one tqdm bar per worker plus an overall batch bar on the terminal, plain lines in run.log, and
every event but progress in events.jsonl."""

import dataclasses
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from resonance.log import style
from resonance.log.events import Event

HEARTBEAT = 60.0            # seconds between progress lines in run.log, where no bar is visible


class Console:
    def __init__(self, run_dir: Path | None = None, total_batches: int = 0) -> None:
        self.bars: dict[int | None, tqdm] = {}
        self.beats: dict[int | None, list] = {}
        self.interactive = sys.stderr.isatty()
        self.log = self.events = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log = open(run_dir / "run.log", "a")
            self.events = open(run_dir / "events.jsonl", "a")
        self.overall = (tqdm(total=total_batches, desc="batches", unit="batch", position=0, leave=True,
                             disable=not self.interactive, dynamic_ncols=True) if total_batches else None)

    def say(self, text: str) -> None:
        """A line on the terminal and, stripped of color, in run.log."""

        tqdm.write(text, file=sys.stdout)
        sys.stdout.flush()
        if self.log:
            for line in style.plain(text).splitlines():
                self.log.write(f"{time.strftime('%F %T')} {line}\n" if line.strip() else "\n")
            self.log.flush()

    def handle(self, event: Event) -> None:
        if event.kind == "progress":
            self._progress(event)
            return
        if self.events:
            self.events.write(json.dumps(dataclasses.asdict(event)) + "\n")
            self.events.flush()
        where = " ".join(part for part in (self._tag(event), event.batch, event.stage) if part)
        if event.kind == "stage_start":
            self._close(event.worker)
            self.say(f"{style.DIM}  · {where} started{style.RESET}")
        elif event.kind == "stage_done":
            self._close(event.worker)
            self.say(style.done(f"{where}  {style.summary(event.summary)}"))
        elif event.kind == "stage_failed":
            self._close(event.worker)
            self.say(style.error(f"{where} FAILED\n{event.message}"))
        elif event.kind == "batch_done":
            if self.overall is not None:
                self.overall.update(1)
        elif event.kind == "batch_failed":
            self.say(style.error(f"{where} stopped: {event.message}"))
        elif event.kind == "warning":
            self.say(style.warning(f"{where + ': ' if where else ''}{event.message}"))
        elif event.kind == "error":
            self.say(style.error(f"{where + ': ' if where else ''}{event.message}"))
        else:
            self.say(f"  {where + ': ' if where else ''}{event.message}")

    def close(self) -> None:
        for worker in list(self.bars):
            self._close(worker)
        if self.overall is not None:
            self.overall.close()
        for handle in (self.log, self.events):
            if handle:
                handle.close()

    @staticmethod
    def _tag(event: Event) -> str:
        return "" if event.worker is None else f"[w{event.worker}]"

    def _progress(self, event: Event) -> None:
        bar = self.bars.get(event.worker)
        if bar is None:
            label = " ".join(filter(None, (self._tag(event), event.batch, event.stage)))
            bar = self.bars[event.worker] = tqdm(total=event.total, desc=label, position=(event.worker or 0) + 1,
                                                 leave=False, disable=not self.interactive, dynamic_ncols=True)
            self.beats[event.worker] = [label, 0, time.time()]
        bar.update(event.advance)
        beat = self.beats[event.worker]         # counted here: a disabled bar keeps no state
        beat[1] += event.advance
        if self.log and time.time() - beat[2] >= HEARTBEAT:
            beat[2] = time.time()
            self.log.write(f"{time.strftime('%F %T')}   … {beat[0]} {beat[1]}/{event.total}\n")
            self.log.flush()

    def _close(self, worker: int | None) -> None:
        bar = self.bars.pop(worker, None)
        if bar is not None:
            bar.close()
