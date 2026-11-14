"""The contract every pipeline module fulfils."""

import abc
import dataclasses
import re
from pathlib import Path
from typing import Any, ClassVar

from resonance.config.schema import Config
from resonance.log.events import Reporter
from resonance.pipeline.artifacts import BatchRef, Layout


FATAL = re.compile(r"401|403|unauthorized|gated|cuda|out of memory", re.IGNORECASE)


class FatalError(RuntimeError):
    """A failure that makes every further clip pointless: bad credentials, a broken GPU, out of memory."""


def guard(error: str, errors: int, total: int, limit: float) -> None:
    """Called for every clip that raised: stop the stage at once on a fatal error, or when more than `limit` of
    the stage's clips have failed. Errors are never counted as rejections."""

    if FATAL.search(error):
        raise FatalError(f"fatal error, stopping the stage: {error}")
    if errors > limit * total:
        raise FatalError(f"{errors} of {total} clips failed, over the limit of {limit:.1%}; last: {error}")


@dataclasses.dataclass
class Context:
    config: Config
    layout: Layout
    work_dir: Path                      # the batch's folder for a batch stage, the run's folder for a pool stage
    log: Reporter
    batch: BatchRef | None = None
    gpus: list[int | None] = dataclasses.field(default_factory=lambda: [None])
    capture: bool = False               # in a worker: the stage's raw output goes to logs/<stage>.log
    summary: dict = dataclasses.field(default_factory=dict)     # filled by the stage, saved with its done marker


class Stage(abc.ABC):
    """One step of a pipeline: a well-specified input artifact in, a well-specified output artifact out.

    A `batch` stage runs once per batch, a `pool` stage once per run over all batches together. `provides` names
    side files the stage leaves in its work folder, `needs` the side files it reads there; the runner checks both,
    and the artifact types, before any work starts.
    """

    name: ClassVar[str]
    scope: ClassVar[str]                # "batch" or "pool"
    consumes: ClassVar[Any]
    produces: ClassVar[Any]
    needs_gpu: ClassVar[bool] = False
    provides: ClassVar[tuple[str, ...]] = ()
    needs: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def enabled(cls, config: Config) -> bool:
        return True

    @abc.abstractmethod
    def output(self, ctx: Context) -> Any:
        """Where this stage's output lives, without running it."""

    @abc.abstractmethod
    def run(self, source: Any, ctx: Context) -> Any:
        """Produce the output from `source`; put the numbers worth reporting in ctx.summary."""

    def log_path(self, ctx: Context) -> Path:
        """Where the raw output of this stage's processes goes."""

        return ctx.work_dir / "logs" / f"{self.name}.log"

    def cleanup(self, source: Any, ctx: Context) -> None:
        """Called after the done marker is written, for work that must not be lost to a crash before it."""
