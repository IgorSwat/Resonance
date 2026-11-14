"""Sends a process's raw output to a file, so library chatter never reaches the console."""

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def captured(path: Path | None) -> Iterator[None]:
    """Inside the block, stdout and stderr of this process and of every process it starts are appended to `path`.

    Done on the file descriptors, not on sys.stdout, so C libraries and child processes are covered too.
    """

    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for descriptor, target in zip(saved, (1, 2)):
                os.dup2(descriptor, target)
                os.close(descriptor)
