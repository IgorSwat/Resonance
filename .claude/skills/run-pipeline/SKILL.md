---
name: run-pipeline
description: Run the Resonance YODAS2-Sidon pipeline on a GPU box — set up the box, configure the batches, launch, monitor, resume after a failure, and check the output. Use when the user wants more data processed.
---

# Run the pipeline

A run takes hours and fills disks. Prepare the box, check the plan with a dry run, launch inside `tmux`, and verify
the output before reporting it.

## 1. Plan the run

State to the user before starting:

- **Batches:** language, shard and range. A shard's batch count differs: English has 34 shards of about 130
  batches, Polish has one of 38. The preflight check reports each shard's range.
- **Yield:** one English batch gives about 7,300 clips (20 h) after segmenting, and the filters keep about half.
- **Disk:** about 1.75 GB per batch for good (clips kept after `discard_rejected`), the same again while
  `finalize` writes the zip, and up to 3.5 GB per batch in flight. `--dry-run` prints the estimate and warns when
  the batches don't fit.
- **Time:** measured figures are in `knowledge/yodas2_sidon.md` §16.

## 2. Prepare the box

Ask the user which box to use. Then, in the checkout:

```bash
git pull
bash scripts/setup.sh         # asks for the HF token once, installs aria2 and the package, caches the models
```

`setup.sh` ends with the preflight table. Every row must be `ok` or `warning` before a run. The HF account behind
the token must have accepted the terms of `pyannote/segmentation-3.0`.

## 3. Configure and launch

Copy `configs/yodas2_sidon.yaml`, set `run.name`, `run.data_root` and `source.batches`, and leave the calibrated
thresholds alone unless the user asks. Then:

```bash
python scripts/run_pipeline.py --config <config> --dry-run      # plan, preflight, disk estimate, what is done
tmux new -s run
python scripts/run_pipeline.py --config <config>
```

One key can be overridden without editing the file: `--set resources.gpus=[0]`. `--until <stage>` stops early;
`--from <stage>` reruns a stage and everything after it.

## 4. Monitor

- The terminal shows one progress bar per GPU and a line per finished stage with its counts.
- `python scripts/status.py --config <config>` prints clips and hours out of every stage for every batch.
- `<data_root>/runs/<run>/run.log` has the same lines as the terminal plus a progress line per minute;
  `events.jsonl` has every event as JSON. Raw library output of a stage is in
  `<data_root>/batches/<shard>/<batch>/logs/<stage>.log`.
- Stop and tell the user when: a stage fails; the disk warning appears; a batch's accepted share is far from half;
  or `errors` in a stage summary is not zero.

## 5. Resume after a failure

Read the error in `run.log` first: rerunning a deterministic failure only repeats it. Then run the same command
again. Finished stages are skipped by their done markers, per batch.

## 6. Check the output

- `<data_root>/runs/<run>/dataset/` holds `<run>.csv` and `audio.zip` (and `codecs/<codec>/` if configured). The
  CSV's row count equals the number of files in the zip; `finalize` verifies the zip before removing `audio/`.
- Hours, speakers and the accepted share are close to the figures in `knowledge/yodas2_sidon.md`; a large gap needs
  an explanation.
- Ask the user to listen to a sample of the selected clips.

## Done

The run is done when `finalize` has finished, the checks pass, and you have reported: the box, the batches, clips
and hours out of each stage, the final hours and speakers, where the dataset lives, and anything that went wrong.
Record anything new or surprising in `knowledge/yodas2_sidon.md`.
