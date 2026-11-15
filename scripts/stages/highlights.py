"""
Pick a run's most expressive, most aroused and most unusual-voiced clips into runs/<run>/highlights/.

    python scripts/stages/highlights.py --config configs/yodas2_sidon.yaml
    python scripts/stages/highlights.py --config configs/yodas2_sidon.yaml --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("highlights", __doc__)
