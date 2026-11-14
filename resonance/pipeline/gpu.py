"""Processes pinned to one GPU each, and work split across them."""

import contextlib
import multiprocessing
import os
import resource
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from resonance.log.capture import captured
from resonance.log.events import Progress

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def discover(requested: str | list[int]) -> list[int | None]:
    """GPU indices to use; [None] on a machine without NVIDIA GPUs, meaning one worker on whatever torch finds.

    Asked of nvidia-smi rather than torch, so the parent process never initialises CUDA.
    """

    if not shutil.which("nvidia-smi"):
        return [None]
    listed = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.split()
    available = [int(index) for index in listed]
    if requested == "all":
        return available or [None]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"resources.gpus names GPU(s) {missing}, but nvidia-smi lists {available}")
    return list(requested)


def raise_file_limit() -> None:
    """Worker pools and thousands of clips need more than the default 1024 open files."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 1_048_576 if hard == resource.RLIM_INFINITY else hard
    if soft != resource.RLIM_INFINITY and soft < target:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


@contextlib.contextmanager
def pinned(gpu: int | None, threads: int) -> Iterator[None]:
    """The environment a process started inside this block inherits: one visible GPU and capped thread pools.

    The caps matter on many-core boxes: every BLAS and OpenMP pool otherwise starts one thread per core in every
    worker, which runs into the container's process limit.
    """

    env = {name: str(threads) for name in THREAD_VARIABLES}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def spawn(target: Callable, args: tuple, gpu: int | None, threads: int) -> multiprocessing.Process:
    """Spawned, never forked: a child forked after CUDA is initialised inherits a context it cannot use."""

    with pinned(gpu, threads):
        process = multiprocessing.get_context("spawn").Process(target=target, args=args)
        process.start()
    return process


def map_gpus(function: Callable[[Any, Callable[[int], None]], Any], shards: Sequence[Any],
             gpus: Sequence[int | None], threads: int, progress: Progress | None = None,
             log: Path | None = None) -> list:
    """function(shard, advance) for every shard, each in its own process on its own GPU; results in shard order.

    Always a child process, even for one shard, so the caller never holds CUDA or a model's memory. The
    children's raw output is appended to `log`.
    """

    results = multiprocessing.get_context("spawn").Queue()
    processes = [spawn(_shard, (function, index, shard, results, log), gpus[index % len(gpus)], threads)
                 for index, shard in enumerate(shards)]
    out: dict[int, Any] = {}
    try:
        while len(out) < len(shards):
            kind, index, payload = results.get()
            if kind == "progress":
                if progress is not None:
                    progress.advance(payload)
            elif kind == "error":
                raise RuntimeError(f"shard {index} failed:\n{payload}")
            else:
                out[index] = payload
    finally:
        for process in processes:
            if len(out) < len(shards):
                process.terminate()
            process.join()
    if progress is not None:
        progress.flush()
    return [out[index] for index in range(len(shards))]


def _shard(function: Callable, index: int, shard: Any, results: Any, log: Path | None) -> None:
    try:
        with captured(log):
            value = function(shard, lambda count=1: results.put(("progress", index, count)))
        results.put(("done", index, value))
    except Exception:
        results.put(("error", index, traceback.format_exc()))


def split(items: Sequence[Any], parts: int) -> list[list[Any]]:
    """`items` dealt into at most `parts` non-empty shards of near-equal size."""

    parts = max(1, min(parts, len(items)))
    return [list(items[index::parts]) for index in range(parts)]
