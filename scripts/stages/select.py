"""Select clips from a run's pool within the per-speaker bounds.

    python scripts/stages/select.py --config configs/yodas2_sidon.yaml
    python scripts/stages/select.py --config configs/yodas2_sidon.yaml --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("select", __doc__)
