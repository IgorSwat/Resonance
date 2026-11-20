"""
Runs a whole language unattended: chunk after chunk, the pipeline as a watched child process, each finished chunk
uploaded to Drive in the background and deleted locally once its hashes check out.

The pipeline is a child process so that a crash, an out-of-memory kill or a hang ends only the child: it is
started again with the same arguments and resumes from its done markers. A batch that keeps failing is left out of
its chunk; a pool stage that keeps failing, or an upload that does, stops everything for a human to look at.
State lives in <data_root>/orchestrator/<language>.json, and the chunks already on Drive are skipped, so the
orchestrator itself can be restarted at any time.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from resonance.config import loader
from resonance.config.schema import Config
from resonance.orchestrate import drive
from resonance.orchestrate.chunks import Chunk, plan
from resonance.pipeline import state
from resonance.pipeline.artifacts import BatchRef, Layout

BACKOFF = (60, 300, 900)        # seconds before the 1st, 2nd and later restarts
PIPELINE = "from resonance.pipeline.cli import main_pipeline; main_pipeline('Resonance pipeline')"


class Stop(RuntimeError):
    """Something a restart cannot fix; the orchestrator stops and keeps everything on disk."""


class Ledger:
    """Per-chunk state: running → finished → uploaded → done, with failure counts and quarantined batches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.chunks: dict[str, dict[str, Any]] = json.loads(path.read_text())["chunks"] if path.exists() else {}
        self.lock = threading.Lock()

    def entry(self, chunk: Chunk) -> dict[str, Any]:
        return self.chunks.setdefault(chunk.name, {"state": "pending", "skipped": [], "failures": {},
                                                   "pool_failures": 0, "other_failures": 0, "restarts": 0})

    def update(self, chunk: Chunk, **fields: Any) -> None:
        with self.lock:
            self.entry(chunk).update(fields)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"chunks": self.chunks}, indent=1))
            os.replace(temporary, self.path)


class Orchestrator:
    def __init__(self, config_path: Path, overrides: list[str], language: str, start: BatchRef | None = None,
                 shards: list[str] | None = None, command: Callable[[Chunk, bool], list[str]] | None = None) -> None:
        self.config_path, self.overrides, self.language, self.start = config_path, overrides, language, start
        self.base = loader.load(config_path, overrides)
        self.settings = self.base.orchestrate
        self.shards = shards if shards is not None else self.settings.shards
        self.home = self.base.run.data_root / "orchestrator"
        self.ledger = Ledger(self.home / f"{language}.json")
        self.command = command or self.pipeline_command
        self.upload_thread: threading.Thread | None = None
        self.upload_error: BaseException | None = None
        self.log_lock = threading.Lock()

    # ---- logging

    def log(self, message: str) -> None:
        line = f"{time.strftime('%F %T')} {message}"
        with self.log_lock:
            print(line, flush=True)
            self.home.mkdir(parents=True, exist_ok=True)
            with open(self.home / f"{self.language}.log", "a") as handle:
                handle.write(line + "\n")

    # ---- the plan

    def chunk_overrides(self, chunk: Chunk) -> list[str]:
        batches = ", ".join(f"{{shard: '{span.shard}', first: {span.first}, last: {span.last}}}"
                            for span in chunk.ranges())
        return [*self.overrides, f"run.name={chunk.name}", f"source.language={chunk.language}",
                f"source.batches=[{batches}]"]

    def chunk_config(self, chunk: Chunk) -> Config:
        return loader.load(self.config_path, self.chunk_overrides(chunk))

    def pipeline_command(self, chunk: Chunk, preflight: bool) -> list[str]:
        sets = [part for override in self.chunk_overrides(chunk) for part in ("--set", override)]
        return [sys.executable, "-c", PIPELINE, "--config", str(self.config_path), *sets,
                *([] if preflight else ["--skip-preflight"])]

    def todo(self) -> list[Chunk]:
        """The chunks still to run, upload or clean, in order; the ones already on Drive are marked done."""

        chunks = plan(self.language, self.settings.chunk_batches, self.shards, self.start)
        on_drive = drive.uploaded(self.settings.upload, self.language, [chunk.name for chunk in chunks])
        result = []
        for chunk in chunks:
            entry = self.ledger.entry(chunk)
            chunk.skipped = list(entry["skipped"])
            if chunk.name in on_drive and entry["state"] in ("pending", "done"):
                if entry["state"] != "done":
                    self.ledger.update(chunk, state="done", note="already on Drive")
                continue
            if entry["state"] != "done":
                result.append(chunk)
        return result

    # ---- the loop

    def run(self) -> None:
        chunks = self.todo()
        self.log(f"orchestrator {self.language}: {len(chunks)} chunk(s) to do"
                 + (f", {chunks[0].name} … {chunks[-1].name}" if chunks else ""))
        try:
            for chunk in chunks:
                if self.ledger.entry(chunk)["state"] in ("pending", "running"):
                    self.wait_for_disk()
                    self.run_chunk(chunk)
                self.join_upload()
                self.upload_thread = threading.Thread(target=self.deliver, args=(chunk,), daemon=True)
                self.upload_thread.start()
            self.join_upload()
        except (Stop, KeyboardInterrupt) as error:
            self.log(f"STOPPED: {error or 'interrupted'}")
            if self.upload_thread is not None and self.upload_thread.is_alive():
                self.log("waiting for the upload in progress to finish")
                self.upload_thread.join()
            raise
        self.log(f"orchestrator {self.language}: every chunk is done")

    def join_upload(self) -> None:
        if self.upload_thread is not None:
            self.upload_thread.join()
            self.upload_thread = None
        if self.upload_error is not None:
            raise Stop(f"upload failed: {self.upload_error}")

    def wait_for_disk(self) -> None:
        needed = self.settings.min_free_gb * 1e9
        self.base.run.data_root.mkdir(parents=True, exist_ok=True)
        while (free := shutil.disk_usage(self.base.run.data_root).free) < needed:
            if self.upload_thread is not None:
                self.log(f"disk: {free / 1e9:.0f} GB free, below {self.settings.min_free_gb:.0f}; waiting for the upload")
                self.join_upload()
                continue
            self.log(f"disk: {free / 1e9:.0f} GB free, below {self.settings.min_free_gb:.0f}; waiting for space")
            time.sleep(self.settings.poll_seconds)

    # ---- one chunk through the pipeline

    def run_chunk(self, chunk: Chunk) -> None:
        layout = Layout(self.chunk_config(chunk))
        attempt = 0
        while True:
            entry = self.ledger.entry(chunk)
            self.ledger.update(chunk, state="running", started=entry.get("started") or time.time())
            outcome, open_batches, failed, pool_failed = self.attempt(chunk, layout, preflight=attempt == 0)
            if outcome == 0 and state.done(layout.run_dir, "finalize"):
                summary = state.summary(layout.run_dir, "finalize") or {}
                self.ledger.update(chunk, state="finished", finished=time.time(), hours=summary.get("hours_out"),
                                   clips=summary.get("clips_out"), audio_gb=summary.get("audio_gb"))
                self.log(f"{chunk.name}: finished, {summary.get('clips_out', 0):,} clips, "
                         f"{summary.get('hours_out', 0):.1f} h, {summary.get('audio_gb', 0):.1f} GB")
                return
            self.record_failure(chunk, outcome, failed or (open_batches if outcome == "stalled" else []), pool_failed)
            attempt += 1
            wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            self.ledger.update(chunk, restarts=self.ledger.entry(chunk)["restarts"] + 1)
            self.log(f"{chunk.name}: restarting in {wait / 60:.0f} min (restart {attempt})")
            time.sleep(wait)

    def attempt(self, chunk: Chunk, layout: Layout, preflight: bool) -> tuple[int | str, list[str], list[str], bool]:
        """One run of the child. Returns (exit code or 'stalled', batches in flight, failed batches, pool failed)."""

        events_path = layout.run_dir / "events.jsonl"
        offset = events_path.stat().st_size if events_path.exists() else 0
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self.home / f"{chunk.name}.out", "a") as output:
            process = subprocess.Popen(self.command(chunk, preflight), stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        started = time.time()
        self.log(f"{chunk.name}: pipeline started (pid {process.pid}), batches "
                 + ", ".join(f"{span.shard}/{span.first}-{span.last}" for span in chunk.ranges()))
        outcome: int | str
        while True:
            code = process.poll()
            if code is not None:
                outcome = code
                break
            log_file = layout.run_dir / "run.log"
            active = max(started, log_file.stat().st_mtime if log_file.exists() else 0)
            if time.time() - active > self.settings.stall_minutes * 60:
                self.log(f"{chunk.name}: run.log silent for {self.settings.stall_minutes} min; killing pid {process.pid}")
                kill_tree(process.pid)
                outcome = "stalled"
                break
            time.sleep(self.settings.poll_seconds)
        open_batches, failed, pool_failed = read_events(events_path, offset)
        if outcome != 0:
            self.log(f"{chunk.name}: pipeline ended ({outcome})"
                     + (f", failed: {', '.join(failed)}" if failed else "")
                     + (", a pool stage failed" if pool_failed else ""))
        return outcome, open_batches, failed, pool_failed

    def record_failure(self, chunk: Chunk, outcome: int | str, batches: list[str], pool_failed: bool) -> None:
        entry, limit = self.ledger.entry(chunk), self.settings.max_failures
        if pool_failed:
            count = entry["pool_failures"] + 1
            self.ledger.update(chunk, pool_failures=count)
            if count >= limit:
                raise Stop(f"{chunk.name}: a pool stage failed {count} times; see {chunk.name}'s run.log")
            return
        if not batches:
            count = entry["other_failures"] + 1
            self.ledger.update(chunk, other_failures=count)
            if count >= limit:
                raise Stop(f"{chunk.name}: the pipeline ended ({outcome}) {count} times without a failed batch; "
                           f"see {self.home / (chunk.name + '.out')}")
            return
        failures = dict(entry["failures"])
        for key in batches:
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= limit:
                index = BatchRef.parse(key).index
                if index not in chunk.skipped:
                    chunk.skipped.append(index)
                    self.log(f"{chunk.name}: {key} failed {failures[key]} times; quarantined, the chunk goes on "
                             "without it")
        if not chunk.ranges():
            raise Stop(f"{chunk.name}: every batch is quarantined")
        self.ledger.update(chunk, failures=failures, skipped=sorted(chunk.skipped))

    # ---- upload and cleanup, on the background thread

    def deliver(self, chunk: Chunk) -> None:
        try:
            layout = Layout(self.chunk_config(chunk))
            if self.ledger.entry(chunk)["state"] == "finished":
                self.upload(chunk, layout)
            self.cleanup(chunk, layout)
        except BaseException as error:      # noqa: BLE001 - reported to the main thread
            self.upload_error = error

    def upload(self, chunk: Chunk, layout: Layout) -> None:
        settings = self.settings.upload
        for attempt in range(1, settings.retries + 1):
            try:
                drive.upload(settings, chunk.language, chunk.name, layout.dataset, layout.highlights, self.log)
                self.ledger.update(chunk, state="uploaded", uploaded=time.time())
                return
            except drive.UploadError as error:
                self.log(f"upload {chunk.name}: attempt {attempt} failed: {error}")
                if attempt == settings.retries:
                    raise
                time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])

    def cleanup(self, chunk: Chunk, layout: Layout) -> None:
        """Delete what the chunk left locally: the dataset, the highlights, and the clips select left out."""

        for folder in (layout.dataset, layout.highlights):
            shutil.rmtree(folder, ignore_errors=True)
        for index in range(chunk.first, chunk.last + 1):
            ref = BatchRef(chunk.language, chunk.shard, index)
            shutil.rmtree(layout.clips(ref), ignore_errors=True)
            shutil.rmtree(layout.raw(ref), ignore_errors=True)
        self.ledger.update(chunk, state="done", done=time.time())
        self.log(f"{chunk.name}: done, local copy removed")

    # ---- status

    def status(self) -> None:
        rows, uploaded_hours, day_hours = [], 0.0, 0.0
        for name, entry in self.ledger.chunks.items():
            hours = entry.get("hours") or 0.0
            if entry["state"] == "done" and hours:
                uploaded_hours += hours
                if time.time() - entry.get("uploaded", 0) < 86400:
                    day_hours += hours
            rows.append(f"  {name:12s} {entry['state']:9s} {hours:8.1f} h {entry.get('audio_gb') or 0:7.1f} GB "
                        f"restarts {entry['restarts']}" + (f", quarantined {entry['skipped']}" if entry["skipped"] else "")
                        + (f" ({entry['note']})" if entry.get("note") else ""))
        print(f"orchestrator {self.language}: {len(rows)} chunk(s) in {self.ledger.path}")
        print("\n".join(rows))
        print(f"uploaded by the orchestrator: {uploaded_hours:.1f} h in total, {day_hours:.1f} h in the last 24 h")


def read_events(path: Path, offset: int) -> tuple[list[str], list[str], bool]:
    """From events.jsonl past `offset`: (batches with a stage still open, failed batches, whether a pool stage failed)."""

    if not path.exists():
        return [], [], False
    open_stages: dict[str, int] = {}
    failed: list[str] = []
    pool_failed = False
    with open(path) as handle:
        handle.seek(offset)
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch, kind = event.get("batch"), event.get("kind")
            if kind == "stage_start" and batch:
                open_stages[batch] = open_stages.get(batch, 0) + 1
            elif kind in ("stage_done", "stage_failed") and batch:
                open_stages[batch] = max(0, open_stages.get(batch, 0) - 1)
            if kind in ("stage_failed", "batch_failed"):
                if batch:
                    if batch not in failed:
                        failed.append(batch)
                elif event.get("stage"):
                    pool_failed = True
    return [batch for batch, count in open_stages.items() if count and batch not in failed], failed, pool_failed


def kill_tree(pid: int, grace: float = 30.0) -> None:
    """Terminate a process and every descendant, including the ones in their own process groups."""

    listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        child, parent = map(int, line.split())
        children.setdefault(parent, []).append(child)
    tree, queue = [], [pid]
    while queue:
        current = queue.pop()
        tree.append(current)
        queue.extend(children.get(current, []))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for target in tree:
            try:
                os.kill(target, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + grace
        while time.time() < deadline and any(_alive(target) for target in tree):
            time.sleep(0.5)
        if not any(_alive(target) for target in tree):
            return


def _alive(pid: int) -> bool:
    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        if finished:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main(doc: str) -> None:
    from resonance.pipeline.cli import load, parser

    args = parser(doc)
    args.add_argument("--language", required=True, help="the language to process, e.g. en")
    args.add_argument("--start", type=BatchRef.parse, metavar="BATCH",
                      help="first batch to process, e.g. en004/100; everything before it is skipped")
    args.add_argument("--shards", help="only these shards, e.g. 004,005 (default: orchestrate.shards, or all)")
    args.add_argument("--dry-run", action="store_true", help="print the chunks still to do, then stop")
    args.add_argument("--status", action="store_true", help="print the ledger, then stop")
    args = args.parse_args()
    load(args, require_run=False)        # reports a bad config before anything starts
    orchestrator = Orchestrator(args.config, args.overrides, args.language, args.start,
                                args.shards.split(",") if args.shards else None)
    if args.status:
        orchestrator.status()
        return
    if args.dry_run:
        for chunk in orchestrator.todo():
            entry = orchestrator.ledger.entry(chunk)
            print(f"  {chunk.name:12s} batches {chunk.shard}/{chunk.first}-{chunk.last}  {entry['state']}")
        return
    try:
        orchestrator.run()
    except Stop:
        sys.exit(1)
