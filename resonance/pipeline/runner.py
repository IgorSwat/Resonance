"""Runs a pipeline: checks the stage chain, resumes from done markers, schedules batches, then runs pool stages."""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from resonance.config import loader
from resonance.config.schema import Config
from resonance.log.capture import captured
from resonance.log.console import Console
from resonance.log.events import Reporter
from resonance.pipeline import gpu, registry, state
from resonance.pipeline.artifacts import BatchRef, Layout, batches
from resonance.pipeline.stage import Context, Stage


class PipelineError(ValueError):
    pass


def resolve(config: Config, classes: Sequence[type[Stage]] | None = None) -> list[type[Stage]]:
    """The enabled stages of the config's pipeline, in order."""

    classes = classes or [registry.get(name) for name in config.pipeline]
    return [cls for cls in classes if cls.enabled(config)]


def validate(stages: Sequence[type[Stage]]) -> tuple[list[type[Stage]], list[type[Stage]]]:
    """(batch stages, pool stages), after checking that every stage's input is what the one before it produces."""

    batch = [cls for cls in stages if cls.scope == "batch"]
    pool = [cls for cls in stages if cls.scope == "pool"]
    if list(stages) != batch + pool:
        raise PipelineError("Batch stages must all come before pool stages: "
                            + " → ".join(f"{cls.name} ({cls.scope})" for cls in stages))
    expected: Any = BatchRef
    for group in (batch, pool):
        provided: set[str] = set()
        for index, cls in enumerate(group):
            wanted = list[expected] if group is pool and index == 0 and batch else expected
            if cls.consumes != wanted:
                raise PipelineError(f"{cls.name} consumes {_name(cls.consumes)}, but it would be given "
                                    f"{_name(wanted)}")
            missing = set(cls.needs) - provided
            if missing:
                raise PipelineError(f"{cls.name} needs {', '.join(sorted(missing))}, which no earlier "
                                    f"{cls.scope} stage provides")
            provided |= set(cls.provides)
            expected = cls.produces
    return batch, pool


def _name(kind: Any) -> str:
    return kind.__name__ if isinstance(kind, type) else str(kind)


def execute(stage: Stage, source: Any, ctx: Context) -> Any:
    """Run one stage, write its done marker with the summary, then let it clean up."""

    ctx.log.batch, ctx.log.stage = ctx.batch.key if ctx.batch else None, stage.name
    ctx.summary = {}
    ctx.log.emit("stage_start")
    started = time.time()
    try:
        with captured(stage.log_path(ctx) if ctx.capture else None):
            result = stage.run(source, ctx)
    except Exception:
        ctx.log.emit("stage_failed", message=traceback.format_exc())
        raise
    summary = {**ctx.summary, "seconds": round(time.time() - started, 1)}
    state.mark(ctx.work_dir, stage.name, summary)
    stage.cleanup(source, ctx)
    ctx.log.emit("stage_done", summary=summary)
    return result


def last_done(stages: Sequence[Stage], ctx: Context) -> int:
    """Index of the last stage with a done marker, or -1."""

    return max((index for index, stage in enumerate(stages) if state.done(ctx.work_dir, stage.name)), default=-1)


def run_chain(stages: Sequence[Stage], source: Any, ctx: Context, stop: int | None = None) -> Any:
    """Run `stages` in order, starting after the last one already done; with `stop`, only stages[:stop]."""

    position = last_done(stages, ctx)
    artifact = stages[position].output(ctx) if position >= 0 else source
    for stage in stages[position + 1:stop]:
        artifact = execute(stage, artifact, ctx)
    return artifact


def reset(stages: Sequence[type[Stage]], start: str, config: Config) -> None:
    """Clear the done markers of `start` and every stage after it, so they run again."""

    names = [cls.name for cls in stages]
    if start not in names:
        raise PipelineError(f"--from {start}: not in the pipeline ({', '.join(names)})")
    layout, later = Layout(config), names[names.index(start):]
    folders = [layout.batch_dir(ref) for ref in batches(config)]
    if any(_consumed_by(folder) == config.run.name for folder in folders):
        raise PipelineError("This run's finalize has moved clips into the dataset; restore them first: "
                            "python scripts/stages/finalize.py --config <config> --restore")
    if "discard_rejected" in names and start not in ("fetch", "segment") and \
            names.index(start) <= names.index("discard_rejected") and any(
            state.done(folder, "discard_rejected") for folder in folders):
        raise PipelineError(f"--from {start}: the audio of rejected clips is already deleted, so the filters "
                            "cannot rerun on it; use --from fetch")
    scopes = {cls.name: cls.scope for cls in stages}
    for name in later:
        for folder in folders if scopes[name] == "batch" else [layout.run_dir]:
            state.clear(folder, name)


def _consumed_by(folder: Path) -> str | None:
    marker = folder / "consumed.json"
    return json.loads(marker.read_text())["run"] if marker.exists() else None


def run(config: Config, console: Console, start: str | None = None, until: str | None = None,
        classes: Sequence[type[Stage]] | None = None) -> bool:
    """The whole pipeline. True when every stage finished."""

    from resonance.pipeline.scheduler import Scheduler

    stages = resolve(config, classes)
    validate(stages)
    names = [cls.name for cls in stages]
    if until is not None:
        if until not in names:
            raise PipelineError(f"--until {until}: not in the pipeline ({', '.join(names)})")
        stages = stages[:names.index(until) + 1]
    if start is not None:
        reset(stages, start, config)
    batch_stages, pool_stages = validate(stages)

    layout, refs = Layout(config), batches(config)
    gpus = gpu.discover(config.resources.gpus)
    gpu.raise_file_limit()
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    (layout.run_dir / "config.yaml").write_text(loader.dump(config))

    if batch_stages and not Scheduler(config, batch_stages, refs, gpus, console).run():
        return False
    if not pool_stages:
        return True
    ctx = Context(config, layout, layout.run_dir, Reporter(console.handle), gpus=gpus)
    source = pool_source(config, batch_stages, refs)
    try:
        run_chain([cls(config) for cls in pool_stages], source, ctx)
    except Exception:
        return False
    return True


def pool_source(config: Config, batch_stages: Sequence[type[Stage]], refs: Sequence[BatchRef]) -> list:
    """What the first pool stage is given: the last batch stage's output for every batch."""

    layout, last = Layout(config), batch_stages[-1](config)
    return [last.output(Context(config, layout, layout.batch_dir(ref), Reporter(lambda event: None), ref))
            for ref in refs]
