"""Show where a run stands: clips and hours out of every stage, per batch and for the pool stages.

    python scripts/status.py --config configs/yodas2_sidon.yaml
    python scripts/status.py data/yodas2_sidon/runs/en001_91
"""

from resonance.pipeline.cli import main_status

if __name__ == "__main__":
    main_status(__doc__)
