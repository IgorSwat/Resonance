"""
Run a whole pipeline, unattended: fetch, segment, filter, fix speakers, select and pack.

    python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml
    python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml --dry-run
    python scripts/run_pipeline.py --config my.yaml --set resources.gpus=[0] --until speaker_drift
    python scripts/run_pipeline.py --config my.yaml --from select

Every batch runs on whichever GPU is free. The run is resumable: start the same command again and finished
stages are skipped. Progress shows as one bar per GPU; the same information goes to
<data_root>/runs/<run name>/run.log, and `python scripts/status.py` prints where every batch stands.
Start long runs inside tmux.
"""

from resonance.pipeline.cli import main_pipeline

if __name__ == "__main__":
    main_pipeline(__doc__)
