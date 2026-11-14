"""Delete the audio of every clip the filters rejected in one batch.

    python scripts/stages/discard_rejected.py --config configs/yodas2_sidon.yaml --batch en001/91
    python scripts/stages/discard_rejected.py --config configs/yodas2_sidon.yaml --batch en001/91 --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("discard_rejected", __doc__)
