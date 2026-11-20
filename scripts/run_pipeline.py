"""
Run one pipeline run: fetch, segment, filter, fix speakers, select and pack. scripts/run_language.py runs these
for a whole language; this runs one by hand, naming its batches with --set (or in a copy of the config).

    RUN=(--set run.name=en004_100 --set source.language=en --set "source.batches=[{shard: '004', first: 50, last: 99}]")
    python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml "${RUN[@]}"
    python scripts/run_pipeline.py --config configs/yodas2_sidon.yaml "${RUN[@]}" --dry-run
    python scripts/run_pipeline.py --config my.yaml --set resources.gpus=[0] --until speaker_drift
    python scripts/run_pipeline.py --config my.yaml --from select

Every batch runs on whichever GPU is free. The run is resumable: start the same command again and finished
stages are skipped. Progress shows as one bar per GPU; the same information goes to
<data_root>/runs/<run name>/run.log, and `python scripts/status.py` prints where every batch stands.
Start long runs inside tmux.
"""

from resonance.pipeline.cli import main_pipeline

if __name__ == "__main__":
    main_pipeline(__doc__)
