"""Run the quality cascade over one segmented batch; writes quality.csv and verdicts.csv.

    python scripts/stages/quality.py --config configs/yodas2_sidon.yaml --batch en001/91
    python scripts/stages/quality.py --config configs/yodas2_sidon.yaml --batch en001/91 --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("quality", __doc__)
