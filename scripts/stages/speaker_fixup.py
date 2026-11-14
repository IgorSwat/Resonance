"""
Merge the video IDs of a run's pool that are one person.

    python scripts/stages/speaker_fixup.py --config configs/yodas2_sidon.yaml
    python scripts/stages/speaker_fixup.py --config configs/yodas2_sidon.yaml --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("speaker_fixup", __doc__)
