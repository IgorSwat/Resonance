"""
Add the normalized transcript to a run's pool, and drop clips whose numbers Whisper and Verbalis read differently.

    python scripts/stages/transcript_fixup.py --config configs/yodas2_sidon.yaml
    python scripts/stages/transcript_fixup.py --config configs/yodas2_sidon.yaml --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("transcript_fixup", __doc__)
