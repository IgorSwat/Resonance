from pathlib import Path

import fakes
from resonance.pipeline import disk


def test_hundred_batches_do_not_fit_256_gb(tmp_path: Path) -> None:
    config = fakes.config(tmp_path, 0, 99)
    result = disk.estimate(config, pending=100, total=100, gpus=2, free_gb=256.0)
    # 100 x 3.5 GB x 0.5 kept, twice (clips and zip), plus raw batches and unfiltered clips in flight
    assert 350 < result.needed_gb < 380 and result.needed_gb > result.free_gb
    assert 60 <= result.fits <= 70


def test_small_run_fits_and_options_change_the_estimate(tmp_path: Path) -> None:
    config = fakes.config(tmp_path, 0, 9)
    base = disk.estimate(config, 10, 10, 2, 256.0)
    assert base.needed_gb < base.free_gb
    config.finalize.zip = False
    assert disk.estimate(config, 10, 10, 2, 256.0).needed_gb < base.needed_gb
    config.pipeline.remove("discard_rejected")
    assert disk.estimate(config, 10, 10, 2, 256.0).per_batch_gb == config.disk.clips_gb_per_batch
    assert disk.estimate(config, 10, 10, 2, 256.0, measured_gb=1.0).per_batch_gb == 1.0
