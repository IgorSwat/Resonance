"""
Download and unpack one batch of YODAS2-Sidon.

    python scripts/stages/fetch.py --config configs/yodas2_sidon.yaml --batch en001/91
    python scripts/stages/fetch.py --config configs/yodas2_sidon.yaml --batch en001/91 --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("fetch", __doc__)
