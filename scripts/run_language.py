"""
Process a whole language unattended: every batch, in chunks of orchestrate.chunk_batches, each chunk a pipeline
run that is restarted when it fails or hangs, then uploaded to Google Drive and deleted locally.

    python scripts/run_language.py --config configs/yodas2_sidon.yaml --language en --start en004/100
    python scripts/run_language.py --config configs/yodas2_sidon.yaml --language en --dry-run
    python scripts/run_language.py --config configs/yodas2_sidon.yaml --language en --status

Chunks already on Drive are skipped, and the state is kept in <data_root>/orchestrator/<language>.json, so the
same command resumes after a stop. The log is <data_root>/orchestrator/<language>.log. Start it inside tmux.
"""

from resonance.orchestrate.supervisor import main

if __name__ == "__main__":
    main(__doc__)
