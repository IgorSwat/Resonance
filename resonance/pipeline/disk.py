"""The disk a run needs against the disk there is. Only ever a warning: the estimate is rough."""

import dataclasses
import shutil
from pathlib import Path
from typing import Sequence

from resonance.config.schema import Config
from resonance.pipeline import state
from resonance.pipeline.artifacts import BatchRef, Layout

GB = 1024 ** 3


@dataclasses.dataclass
class Estimate:
    needed_gb: float
    free_gb: float
    per_batch_gb: float     # what one more batch costs for good, zip included
    fixed_gb: float         # raw batches and unfiltered clips in flight

    @property
    def fits(self) -> int:
        """How many batches the free space holds."""

        return max(0, int((self.free_gb - self.fixed_gb) / self.per_batch_gb)) if self.per_batch_gb else 0


def estimate(config: Config, pending: int, total: int, gpus: int, free_gb: float,
             measured_gb: float | None = None) -> Estimate:
    """
    Disk still needed for `pending` batches of a run of `total`.

    A batch's clips cost clips_gb_per_batch while it is filtered and kept_share of that afterwards, when the
    pipeline discards rejected audio. Zipping needs the selected clips a second time: at most every kept clip of
    the run. `measured_gb` replaces the configured per-batch figure with what finished batches really take.
    """

    discard = "discard_rejected" in config.pipeline
    full = config.disk.clips_gb_per_batch
    kept = measured_gb if measured_gb is not None else full * (config.disk.kept_share if discard else 1.0)
    zipped = "finalize" in config.pipeline and config.finalize.zip
    fixed = config.disk.raw_gb_per_batch * (config.resources.fetch_ahead + gpus) + (full - kept) * gpus
    per_batch = kept * (2 if zipped else 1)
    needed = fixed + kept * pending + (kept * total if zipped else 0)
    return Estimate(needed, free_gb, per_batch, fixed)


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / GB


def measured_gb(layout: Layout, refs: Sequence[BatchRef], done_stage: str) -> float | None:
    """Average size of the clip folders of batches that are through `done_stage`, or None if there are none."""

    sizes = [sum(entry.stat().st_size for entry in layout.clips(ref).iterdir())
             for ref in refs if state.done(layout.batch_dir(ref), done_stage) and layout.clips(ref).is_dir()]
    return sum(sizes) / len(sizes) / GB if sizes else None


def check(config: Config, layout: Layout, refs: Sequence[BatchRef], pending: int, gpus: int,
          done_stage: str) -> str | None:
    """A warning when the remaining batches do not fit on the disk, else None."""

    result = estimate(config, pending, len(refs), gpus, free_gb(layout.root),
                      measured_gb(layout, refs, done_stage))
    if result.needed_gb <= result.free_gb:
        return None
    return (f"disk: {pending} batches still to process need about {result.needed_gb:.0f} GB, "
            f"but {layout.root} has {result.free_gb:.0f} GB free; about {result.fits} more batches fit")
