"""Check that this machine can run a pipeline: packages, aria2c, GPUs, HF token and gated models, disk.

    python scripts/preflight.py --config configs/yodas2_sidon.yaml
    python scripts/preflight.py --offline
    python scripts/preflight.py --download-models      # also cache every model the pipeline uses
"""

from resonance.pipeline.cli import main_preflight

if __name__ == "__main__":
    main_preflight(__doc__)
