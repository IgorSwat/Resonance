"""Pack a run's selected clips into the dataset: <run>.csv, audio.zip and optional codec tokens.

    python scripts/stages/finalize.py --config configs/yodas2_sidon.yaml
    python scripts/stages/finalize.py --config configs/yodas2_sidon.yaml --force
    python scripts/stages/finalize.py --config configs/yodas2_sidon.yaml --restore

Runs this one stage with the same config as the whole pipeline. The stage before it must already be done.
"""

from resonance.pipeline.cli import main_stage

if __name__ == "__main__":
    main_stage("finalize", __doc__)
