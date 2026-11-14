"""A pool of processes that each hold their models and score one clip at a time."""

import multiprocessing
from typing import Any, Callable, Iterable, Iterator


def imap(function: Callable[[Any], Any], tasks: Iterable[Any], workers: int,
         setup: Callable[..., None], setup_args: tuple) -> Iterator[Any]:
    """function(task) for every task, in order; `setup` runs once in each worker before its first task.

    Results come back in order, and a worker's exception surfaces at that task's position. Spawned, never
    forked: a forked child inherits a CUDA context it cannot use.
    """

    if workers <= 1:
        setup(*setup_args)
        yield from map(function, tasks)
        return
    with multiprocessing.get_context("spawn").Pool(workers, setup, setup_args) as pool:
        yield from pool.imap(function, tasks, chunksize=4)
