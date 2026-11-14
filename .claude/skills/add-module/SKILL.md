---
name: add-module
description: Add a new stage (module) to the Resonance pipeline, or a new quality metric or codec — implement it behind the contract, register it, give it config, test it, and verify it on a GPU box. Use when the user wants a new processing step, filter, scorer, metric, or codec.
---

# Add a pipeline module

Every step of the pipeline is a module with one job, one input type and one output type. Decide first which of the
three extension points fits:

- **A stage** (`resonance/stages/`): a new step in the pipeline, e.g. a scorer or another filter.
- **A quality metric** (`resonance/metrics/`): a new per-clip check inside the `quality` cascade.
- **A codec** (`resonance/codecs/`): a new token format that `finalize` can write.

## A stage

1. **Fix the contract before writing code.** Scope (`batch` runs once per batch, `pool` once per run), what it
   consumes and produces (`ClipManifest` for almost everything), and any side file it needs or provides. If the
   input or the output takes more than a sentence to describe, split the module in two.
2. **Write `resonance/stages/<name>.py`** with one `Stage` subclass (`resonance/pipeline/stage.py`):
   - `output(ctx)` returns where the output lives; `run(source, ctx)` produces it and fills `ctx.summary` with
     `clips_in`, `clips_out`, `hours_out` and whatever else is worth reporting.
   - Read and write manifests with `resonance/io/manifest.py`; keep the extra columns intact.
   - Report progress with `ctx.log.progress(total)`, and individual problems with `ctx.log.error(...)`. Never
     print.
   - Import heavy libraries inside functions, so tests and the parent process never load them.
   - Per-clip GPU work goes through `resonance/pipeline/pool.py` (batch stages) or `gpu.map_gpus` (pool stages).
     Count per-clip exceptions and pass each to `guard(...)`; an exception is never a rejection.
3. **Register it** in `resonance/pipeline/registry.py`, add its section to `resonance/config/schema.py` (defaults
   are the calibrated values) and to `configs/yodas2_sidon.yaml` with a comment per key, and put its name in the
   `pipeline` list where it belongs. The runner checks at startup that its input type matches the stage before it.
4. **Add `scripts/stages/<name>.py`** (copy a neighbour; only the docstring and the name change). Don't give it
   the name of a standard-library module: it would shadow that module for every script in the folder, which is
   why `select` lives in `select_clips.py`.
5. **Test the logic without a GPU** in `tests/`: pure functions directly, and the stage inside the fake pipeline of
   `tests/test_runner.py` when it changes what flows between stages.
6. **Verify on a GPU box** on one batch, and record the measured yield and speed in `knowledge/yodas2_sidon.md`.

## A quality metric

Subclass `Metric` (`resonance/metrics/metric.py`): `evaluate` returns a dict of scores, `passes` compares them with
the bounds. Raise `Unscorable` only when the clip itself cannot be scored. Add a branch to `build()` in
`resonance/metrics/cascade.py`, a config dataclass in the schema, and the name to `quality.metrics` at the position
its cost justifies: cheap metrics first. Calibrate its bounds on real clips before enabling it by default, and
record the calibration in `knowledge/data_filtering.md`.

## A codec

Subclass `Codec` (`resonance/codecs/codec.py`) with a `name` and `encode(audio) -> int16 array (codebooks, frames)`,
and add it to `resonance/codecs/registry.py`. `finalize.codecs: [<name>]` then writes `codecs/<name>/<clip>.npy`.

## Done

The module is done when the pipeline's dry run lists it, the tests pass, a real batch has gone through it on a GPU
box, and its numbers are in the knowledge note.
