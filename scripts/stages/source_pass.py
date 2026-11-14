"""Drop the recordings of one batch that the multi-speaker metric flagged too often.

    python scripts/stages/source_pass.py --config configs/yodas2_sidon.yaml --batch en001/91
    python scripts/stages/source_pass.py --config configs/yodas2_sidon.yaml --batch en001/91 --force

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("source_pass", __doc__)
