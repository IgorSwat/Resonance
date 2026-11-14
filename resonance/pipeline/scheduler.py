"""Runs the batch stages of a pipeline over many batches: one worker process per GPU pulling from a shared queue,
and one downloader thread feeding it."""

import multiprocessing
import queue
import threading
import traceback
from typing import Sequence

from resonance.config.schema import Config
from resonance.log.console import Console
from resonance.log.events import Event, Reporter
from resonance.pipeline import disk, gpu
from resonance.pipeline.artifacts import BatchRef, Layout
from resonance.pipeline.runner import last_done, run_chain
from resonance.pipeline.stage import Context, Stage


class Scheduler:
    def __init__(self, config: Config, stages: Sequence[type[Stage]], refs: Sequence[BatchRef],
                 gpus: Sequence[int | None], console: Console):
        self.config, self.stages, self.refs, self.gpus, self.console = config, list(stages), list(refs), gpus, console
        self.layout = Layout(config)
        # The leading stages that need no GPU (fetch) run in the downloader thread, ahead of the workers.
        self.head = next((index for index, cls in enumerate(self.stages) if cls.needs_gpu), len(self.stages))
        self.slots = threading.Semaphore(max(1, config.resources.fetch_ahead))
        self.waiting: set[str] = set()          # fetched here, not yet through the first GPU stage

    def run(self) -> bool:
        spawn = multiprocessing.get_context("spawn")
        events, tasks, stop = spawn.Queue(), spawn.Queue(), spawn.Event()
        instances = [cls(self.config) for cls in self.stages]
        pending = [ref for ref in self.refs if last_done(instances, self._context(ref)) < len(instances) - 1]
        for _ in range(len(self.refs) - len(pending)):
            self.console.handle(Event("batch_done"))
        if not pending:
            return True

        workers = {slot: gpu.spawn(_worker, (slot, device, self.config, self.stages, tasks, events, stop),
                                   device, self.config.resources.threads_per_worker)
                   for slot, device in enumerate(self.gpus[:len(pending)])}
        feeder = threading.Thread(target=self._feed, args=(pending, instances, tasks, events, stop, len(workers)),
                                  daemon=True)
        feeder.start()

        alive = set(workers)
        while alive:
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                for slot in [slot for slot in alive if not workers[slot].is_alive()]:
                    alive.discard(slot)
                    stop.set()
                    self.console.handle(Event("error", worker=slot, message="worker process died"))
                continue
            if event.kind == "worker_exit":
                alive.discard(event.worker)
                continue
            self.console.handle(event)
            if event.kind == "stage_done" and self.head < len(self.stages) \
                    and event.stage == self.stages[self.head].name and event.batch in self.waiting:
                self.waiting.discard(event.batch)
                self.slots.release()
        for process in workers.values():
            process.join()
        feeder.join(timeout=10)
        return not stop.is_set()

    def _context(self, ref: BatchRef, reporter: Reporter | None = None) -> Context:
        return Context(self.config, self.layout, self.layout.batch_dir(ref),
                       reporter or Reporter(lambda event: None), ref)

    def _feed(self, pending: Sequence[BatchRef], instances: Sequence[Stage], tasks, events, stop,
              workers: int) -> None:
        """Run the head stages for each batch, at most fetch_ahead batches ahead of the GPUs, and queue it."""

        reporter = Reporter(events.put)
        try:
            for position, ref in enumerate(pending):
                ctx = self._context(ref, reporter)
                if last_done(instances, ctx) < self.head - 1:
                    while not self.slots.acquire(timeout=1):
                        if stop.is_set():
                            return
                    if stop.is_set():
                        return
                    warning = disk.check(self.config, self.layout, self.refs, len(pending) - position, len(self.gpus),
                                         self.stages[-1].name)
                    if warning and position % 10 == 0:
                        reporter.batch = reporter.stage = None
                        reporter.warning(warning)
                    self.waiting.add(ref.key)
                    run_chain(instances, ref, ctx, stop=self.head)
                tasks.put(ref)
        except Exception as error:
            stop.set()
            reporter.emit("batch_failed", message=f"{error}\n{traceback.format_exc()}")
        finally:
            for _ in range(workers):
                tasks.put(None)


def _worker(slot: int, device: int | None, config: Config, classes: Sequence[type[Stage]], tasks, events,
            stop) -> None:
    """One GPU's loop: take a batch, run every stage it still needs, report."""

    reporter = Reporter(events.put, worker=slot)
    layout, stages = Layout(config), [cls(config) for cls in classes]
    try:
        while (ref := tasks.get()) is not None:
            if stop.is_set():
                continue
            ctx = Context(config, layout, layout.batch_dir(ref), reporter, ref, [device], capture=True)
            try:
                run_chain(stages, ref, ctx)
                reporter.batch, reporter.stage = ref.key, None
                reporter.emit("batch_done")
            except Exception as error:
                stop.set()
                reporter.emit("batch_failed", message=str(error))
    finally:
        events.put(Event("worker_exit", worker=slot))
