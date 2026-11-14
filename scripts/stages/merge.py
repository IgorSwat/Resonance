"""
Concatenate the filtered batches of a run into one pool.

    python scripts/stages/merge.py --config configs/yodas2_sidon.yaml
    python scripts/stages/merge.py --config configs/yodas2_sidon.yaml --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("merge", __doc__)
