---
name: run-pipeline
description: Run an existing Resonance corpus pipeline end to end on a GPU box — fetch, segment, filter, speaker fixup, select, and finalize — monitor it, resume it after a failure, and check the output. Use when the user wants more data from a corpus Resonance already supports.
---

# Run a corpus pipeline

Corpus runs take hours, fill disks, and fail quietly: a bad worker setting or device turns into silent rejections,
not crashes. Prepare the box, run in stages you can check, and verify the output before reporting it.
[corpora.md](corpora.md) has the command sequence for each supported corpus.

## 1. Plan the run

State to the user before starting:

- **Corpus, language, and range:** which shards or batches, and the expected yield. For example, one English
  YODAS2-Sidon batch gives about 7,300 utterances (19.7 h), and the filter keeps about half.
- **Disk:** raw shards plus outputs (see each corpus in [corpora.md](corpora.md)). Raw downloads are the largest item.
- **Time:** from the throughput figures in [corpora.md](corpora.md).
- **Selection:** prosody selection (default for most corpora), or ΔCE scoring for YODAS2-Sidon, which needs an Echo
  checkout.

## 2. Prepare the box

Ask the user which GPU box to use; it changes often. Then:

```bash
ssh <box>
cd <checkout> && git pull
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h <data disk>
```

- **Environment:** Python 3.12 or later with `requirements.txt` installed. Some stages also need packages that
  `requirements.txt` misses: `faster_whisper` (YODAS2 segmenter) and `omnivoice` (ParlaSpeech filter).
- **Access:** `HF_TOKEN` set, with the licences accepted for Emilia and `pyannote/segmentation-3.0`; `aria2c`
  installed (every fetcher needs it).
- **Device:** run filters, codecs, and segmentation on CUDA. On a Mac, NISQA scores are wrong (set
  `nisqa_device: cpu`), the codec falls back to CPU and produces different tokens, and the segmenter doesn't run.

## 3. Run it

Start long runs inside `tmux`. For a corpus you haven't run on this box before, run the filter on a small sample
first (`--limit 50 --verbose`) and check its verdict breakdown against the corpus note; the box, not the code, is
usually what's wrong.

- **YODAS2-Sidon:** `scripts/pipeline/yodas2_sidon.sh` chains every stage but finalize, with resume markers.
- **Other corpora:** run the stages one by one from [corpora.md](corpora.md), checking each output before the next.

Use the worker counts the scripts default to. More workers on one GPU run out of memory, and out-of-memory errors
inside a metric become silent rejections. Any worker pool that touches CUDA must use a spawn context.

## 4. Monitor

- Filters print progress every 200 clips (`i/N clips, rate, accepted %`) and a verdict breakdown at the end. The
  YODAS2 pipeline logs to `$RUNS/yodas_<lang><shard>_<first>-<last>/run.log`, with one log per step and batch.
- Watch for:
  - repeated `<Metric> failed: …` lines (out of memory, CUDA, or cuFFT errors turning into rejections);
  - `multi_speaker` rejecting nearly everything (a pool forked after CUDA was initialised);
  - an accepted share far below the corpus note's (wrong config, wrong device, or a bad batch);
  - "built-in defaults" in a filter's header (the `--config` path is wrong).
- Stop and tell the user when one of these appears.

## 5. Resume after a failure

- **YODAS2-Sidon:** rerun the same command; finished steps are skipped by their markers. Delete a stale
  `<batch>.fetch_failed` marker first, or the run aborts at once.
- **Other corpora:** rerun from the failed stage; fetchers skip shards already on disk.
- Read the failing step's log before rerunning. Rerunning a deterministic failure only repeats it.

## 6. Check the output

- The selected CSV parses with the `csv` module in the shared dialect (`|`, `QUOTE_NONE`, `escapechar="\\"`), and its
  row count matches the audio files and the `codecs/*.npy` files.
- The finalize report's hours and frame counts match the selection.
- Hours, speakers, and accepted share are close to the corpus note's figures; a large gap needs an explanation.
- Ask the user to listen to a sample of the selected clips.
- Run any `*_prune.py` with `--dry-run` first and show the user what it would delete; it deletes audio, codecs, and
  CSV rows permanently.

## Done

The run is done when the selected subset is finalized in `data/processed/<Corpus>/`, the output checks pass, and
you've reported to the user: the box, the corpus range, the accepted share per stage, the final hours and speakers,
where the output lives, and anything that went wrong. Record anything new or surprising in the corpus's knowledge
note.
