"""
Select clips from a run's pool within the per-speaker bounds.

    python scripts/stages/select_clips.py --config configs/yodas2_sidon.yaml
    python scripts/stages/select_clips.py --config configs/yodas2_sidon.yaml --force

Runs the `select` stage with the same config as the whole pipeline. The stage before it must already be done.
The file is not called select.py: that name would shadow Python's own select module for every script here.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("select", __doc__)
