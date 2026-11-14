"""The command lines behind scripts/: argument parsing and reporting, shared by every script."""

import argparse
import sys
from pathlib import Path

from resonance.config import loader
from resonance.config.schema import Config
from resonance.log import style
from resonance.log.console import Console
from resonance.log.events import Reporter
from resonance.pipeline import preflight, runner, state
from resonance.pipeline.artifacts import BatchRef, Layout, batches
from resonance.pipeline.stage import Context

DEFAULT_CONFIG = Path("configs/yodas2_sidon.yaml")
MARKS = {"ok": f"{style.GREEN}ok{style.RESET}", "warning": f"{style.YELLOW}warning{style.RESET}",
         "fail": f"{style.RED}FAIL{style.RESET}"}


def parser(doc: str) -> argparse.ArgumentParser:
    """An argument parser with the options every script takes; `doc` is the script's docstring."""

    result = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"pipeline config (default: {DEFAULT_CONFIG})")
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override one config key, e.g. --set quality.workers=4 (repeatable)")
    return result


def load(args: argparse.Namespace) -> Config:
    try:
        return loader.load(args.config, args.overrides)
    except loader.ConfigError as error:
        sys.exit(style.error(str(error)))


def report_preflight(config: Config, console: Console, online: bool = True) -> bool:
    """Print the preflight table; False when a check failed."""

    checks = preflight.run(config, online)
    console.say(style.section("Preflight"))
    console.say(style.table(("check", "status", "detail"),
                            [(check.name, MARKS[check.status], check.detail) for check in checks]))
    return all(check.status != "fail" for check in checks)


def main_pipeline(doc: str) -> None:
    args = parser(doc)
    args.add_argument("--from", dest="start", metavar="STAGE",
                      help="rerun this stage and every stage after it, on every batch")
    args.add_argument("--until", metavar="STAGE", help="stop after this stage")
    args.add_argument("--dry-run", action="store_true",
                      help="print the plan, the preflight checks and what is already done, then stop")
    args.add_argument("--skip-preflight", action="store_true", help="start without the preflight checks")
    args = args.parse_args()
    config = load(args)
    try:
        stages = runner.resolve(config)
        runner.validate(stages)
    except (runner.PipelineError, KeyError) as error:
        sys.exit(style.error(str(error.args[0])))

    refs, layout = batches(config), Layout(config)
    console = Console(None if args.dry_run else layout.run_dir, 0 if args.dry_run else len(refs))
    console.say(style.banner(f"Resonance · {config.run.name}"))
    console.say(style.info("config", args.config))
    console.say(style.info("data root", layout.root))
    console.say(style.info("batches", f"{len(refs)} ({refs[0].key} … {refs[-1].key})"))
    console.say(style.info("pipeline", " → ".join(cls.name for cls in stages)))
    ready = args.skip_preflight or report_preflight(config, console)
    if args.dry_run:
        print_status(config, console)
        console.close()
        sys.exit(0 if ready else 1)
    if not ready:
        console.close()
        sys.exit(style.error("Preflight failed; nothing was started."))

    console.say(style.section("Running"))
    try:
        finished = runner.run(config, console, args.start, args.until)
    except runner.PipelineError as error:
        console.close()
        sys.exit(style.error(str(error)))
    print_status(config, console)
    console.say(style.done(f"finished: {layout.run_dir}") if finished
                else style.error(f"stopped with errors; see {layout.run_dir / 'run.log'}. "
                                 "Fix the cause and rerun the same command to resume."))
    console.close()
    sys.exit(0 if finished else 1)


def print_status(config: Config, console: Console) -> None:
    """Batches by stage, then the pool stages, read from the done markers."""

    layout = Layout(config)
    batch_stages, pool_stages = runner.validate(runner.resolve(config))

    def cell(summary: dict | None) -> str:
        if summary is None:
            return "·"
        if "clips_out" in summary:
            return f"{summary['clips_out']:,}" + (f" ({summary['hours_out']:.1f} h)" if "hours_out" in summary else "")
        return "done"

    if batch_stages:
        rows, totals = [], {cls.name: [0, 0.0, 0] for cls in batch_stages}
        for ref in batches(config):
            summaries = [state.summary(layout.batch_dir(ref), cls.name) for cls in batch_stages]
            for cls, summary in zip(batch_stages, summaries):
                if summary:
                    totals[cls.name][0] += summary.get("clips_out", 0)
                    totals[cls.name][1] += summary.get("hours_out", 0.0)
                    totals[cls.name][2] += 1
            last = max((index for index, summary in enumerate(summaries) if summary), default=-1)
            # a stage before the last finished one is done even without a marker: segment removes fetch's
            rows.append((ref.key, *("✓" if summary is None and index < last else cell(summary)
                                    for index, summary in enumerate(summaries))))
        rows.append((f"{style.BOLD}total{style.RESET}", *(
            f"{style.BOLD}{clips:,} ({hours:.1f} h) [{count}]{style.RESET}" if count else "·"
            for clips, hours, count in totals.values())))
        console.say(style.section("Batches (clips out, hours)"))
        console.say(style.table(("batch", *(cls.name for cls in batch_stages)), rows))
    if pool_stages:
        console.say(style.section(f"Run {config.run.name}"))
        console.say(style.table(("stage", "result"), [
            (cls.name, style.summary(summary) if (summary := state.summary(layout.run_dir, cls.name)) else "·")
            for cls in pool_stages]))


def main_status(doc: str) -> None:
    args = parser(doc)
    args.add_argument("run", nargs="?", type=Path,
                      help="a run folder (data root/runs/<name>): its saved config.yaml is used instead of --config")
    args = args.parse_args()
    if args.run is not None:
        args.config = args.run / "config.yaml"
    console = Console()
    print_status(load(args), console)


def main_preflight(doc: str) -> None:
    args = parser(doc)
    args.add_argument("--offline", action="store_true", help="skip the checks that need the network")
    args.add_argument("--download-models", action="store_true",
                      help="after the checks, load every model once so it is cached before the first run")
    args = args.parse_args()
    config, console = load(args), Console()
    ready = report_preflight(config, console, not args.offline)
    if ready and args.download_models:
        console.say(style.section("Models"))
        preflight.download_models(config, lambda line: console.say(f"  {line}"))
    sys.exit(0 if ready else 1)


def main_stage(name: str, doc: str) -> None:
    """Run one stage alone: on one batch for a batch stage, on the run for a pool stage."""

    args = parser(doc)
    args.add_argument("--batch", help="the batch to run a batch stage on, e.g. en001/91")
    args.add_argument("--force", action="store_true", help="run even if the stage is already done")
    if name == "finalize":
        args.add_argument("--restore", action="store_true",
                          help="undo finalize: move the selected clips back and remove the dataset")
    args = args.parse_args()
    config = load(args)
    console = Console()
    if getattr(args, "restore", False):
        from resonance.stages.finalize import restore

        console.say(style.done(f"restored {restore(config):,} clips of run {config.run.name}"))
        return

    try:
        batch_stages, pool_stages = runner.validate(runner.resolve(config))
    except (runner.PipelineError, KeyError) as error:
        sys.exit(style.error(str(error.args[0])))
    layout = Layout(config)
    chain = next((group for group in (batch_stages, pool_stages) if name in [cls.name for cls in group]), None)
    if chain is None:
        sys.exit(style.error(f"{name} is not an enabled stage of this config's pipeline"))
    position = [cls.name for cls in chain].index(name)
    stages = [cls(config) for cls in chain]
    from resonance.pipeline import gpu

    if chain is batch_stages:
        if not args.batch:
            sys.exit(style.error(f"{name} is a batch stage: pass --batch, e.g. --batch en001/91"))
        ref = BatchRef.parse(args.batch)
        ctx = Context(config, layout, layout.batch_dir(ref), Reporter(console.handle), ref)
        source = ref
    else:
        ctx = Context(config, layout, layout.run_dir, Reporter(console.handle),
                      gpus=gpu.discover(config.resources.gpus))
        source = runner.pool_source(config, batch_stages, batches(config)) if batch_stages else None
    if position:
        previous = stages[position - 1]
        if not state.done(ctx.work_dir, previous.name):
            sys.exit(style.error(f"{name} needs the output of {previous.name}, which has not run here yet"))
        source = previous.output(ctx)
    if state.done(ctx.work_dir, name) and not args.force:
        sys.exit(style.warning(f"{name} is already done here; pass --force to run it again"))
    gpu.raise_file_limit()
    try:
        runner.execute(stages[position], source, ctx)
    except Exception:
        sys.exit(1)
    finally:
        console.close()
