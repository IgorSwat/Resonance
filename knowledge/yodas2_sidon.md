# YODAS2-Sidon — what the Spanish batch actually contains

> **2026-11-14:** the pipeline was refactored into the `resonance/` package. §1-§13 are the measurements made with
> the code before that (git tag `pre-refactor`) and name its scripts. They map onto the stages as follows:
> `scripts/segment/yodas2_sidon.py` → `segment`; `scripts/filter/yodas2_sidon.py` → `quality`, `source_pass` and
> `speaker_drift`; `scripts/filter/yodas2_sidon_speaker_fixup.py` → `speaker_fixup`;
> `scripts/finalize/yodas2_sidon_prune.py` is gone, because `speaker_drift` runs on every batch. §14 and §17 were
> moved here from the Emilia note when that corpus was removed. §15 and §16 measure the refactored pipeline.

Measured on `data/YODAS2-sidon/train-00000` (Spanish, one shard of the 1108 under
`sarulab-speech/yodas2_sidon`): 208 videos, 36.7 h, 43,821 utterances, 2.8 GB. Sample sizes are
given because several are small.

Companion notes: `knowledge/data_filtering.md` for the cascade, `knowledge/cml_tts.md` for the
other Spanish source.

---

## 1. The restoration is real

YODAS-2 is 16 kHz. Sidon's output, resampled to 24 kHz, carries genuine content to the full
12 kHz Nyquist: effective bandwidth median 1.00, p10 1.00, with 4% of clips below the 0.85 gate
(23 clips sampled across 25 videos). So the bandwidth stage is **not** the bottleneck here, unlike
CML-TTS, and unlike raw YODAS this needs no OmniVoice regeneration to reach the codec's rate.

**Caveat that no metric in the cascade can see:** everything above ~8 kHz was *synthesised* by the
restoration model, not recorded. `effective_bandwidth` reads it as real. Training a codec or TTS
on hallucinated high frequencies is a risk taken knowingly, not one the pipeline will flag.

## 2. Two thirds of the utterances are too short

Caption segments, not sentences. Median utterance 2.6 s.

| range | utterances | share | hours |
|---|---|---|---|
| 0–1 s | 328 | 0.7% | 0.1 |
| 1–3 s | 28,965 | 66.1% | 17.7 |
| 3–30 s | 14,480 | 33.0% | 17.9 |
| 30 s+ | 48 | 0.1% | 0.6 |

Only 17.9 h of 36.3 h reaches the pipeline's duration window at all. Merging adjacent caption
segments would recover much of the discarded 17.7 h.

## 3. The transcripts are raw auto-captions

Of 43,821 utterance texts:

| property | share |
|---|---|
| ends in `.`, `?` or `!` | 0.0% |
| contains any of `, . ? !` | 0.1% |
| starts with a capital | 1.7% |
| all lowercase | 91.4% |
| carries a `[music]`/`♪`/`aplausos` tag | 3.2% |

No punctuation, no casing, and segments break mid-phrase (`y si me permiten hacer`). For ASR that
is fine; for TTS it means the text carries no prosodic cue at all. Mixing this with a punctuated
corpus teaches a model that punctuation is optional.

## 4. Clip boundaries cut into speech

Of 120 sampled 3–30 s utterances, **48% already have speech running at the first 50 ms** and
**62% still have speech running at the last 50 ms**. The caption timestamps are not utterance
boundaries. This is also most of why the CTC stage rejects so much (§5): a clip that starts
mid-word scores as missing speech against its text.

## 5. Cascade acceptance: 35% of what is eligible

150 random 3–30 s utterances, cut from the flac on their caption timestamps, through
`configurations/quality_filtering_cml_tts.yaml`:

| verdict | clips | share |
|---|---|---|
| accepted | 53 | 35.3% |
| ctc_alignment | 51 | 34.0% |
| nisqa | 27 | 18.0% |
| multi_speaker | 17 | 11.3% |
| spectral_flatness | 2 | 1.3% |

Net yield: **6.3 h of usable audio from 36.7 h of source, 17%.** Across all 1108 Spanish shards
that extrapolates to roughly 7,000 h, at ~2.95 GB downloaded per 6 h kept.

## 6. There are no speaker labels, and video_id is only a rough proxy

The metadata carries `id`, `video_id`, `duration` and `utterances` — nothing about who is
speaking. Grouping by video is the only option, and it is wrong often enough to matter: of 9
videos with at least 8 usable clips, **2 span more than 1.5x in per-clip pitch median**, i.e.
interviews or panel shows. The per-clip `multi_speaker` stage does not catch this, since each clip
is single-voice; it is the same source-level leak as Emilia, and `source_rejection_enabled` in
`scripts/filter/emilia.py` is the existing fix.

## 7. Re-transcribing with Whisper fixes §3 and §4

Tried on 20 videos (5.77 h) of the Spanish batch: `mlx-community/whisper-large-v3-turbo` via
`mlx_whisper` with `word_timestamps=True`, words regrouped into chunks that end on sentence
punctuation, then the normal cascade. Whisper's **own segments are useless** for this (median
1.2 s, 12 of 190 in the duration window) — the word timestamps are what matter.

| | raw captions | Whisper re-cut |
|---|---|---|
| transcripts with any punctuation | 0.1% | 98.6% |
| ending in `.`/`?`/`!` | 0.0% | 98.2% |
| all lowercase | 91.4% | 5.1% |
| carrying a music tag | 3.2% | 0.2% |
| starts mid-speech | 48% | 24% |
| ends mid-speech | 62% | 58%, or 21% with a 0.25 s tail pad |
| cascade acceptance on eligible chunks | 35.3% | 44.5% |

**The tail cuts are not mid-word cuts.** Whisper's last-word `end` is tight, so speech runs to the
final sample. A 0.15 s head / 0.25 s tail pad takes end-mid-speech from 58% to 21%; the same pad
does nothing for the head (24% either way), so those starts are genuinely abrupt.

Rejection shifts with the longer chunks: `multi_speaker` rises to 18.2% from 11.3%, because a
6.4 s median chunk has more chance of catching a second voice than a 2.6 s caption does.

### Cost

Whisper ran at **19.7x realtime** on an M4 Pro (GPU via MLX); the whole pipeline including the
cascade managed 16.9x. **Net yield stays ~18% of source audio** — the same as the caption route —
so the win is entirely in transcript and boundary quality, not volume. Per shard that is roughly
2.2 h of wall time for ~6.6 h kept, and the full Spanish set would be on the order of three
months of single-machine compute.

## 8. Captions must be joined before cutting, and the gap rule is not what does it

`scripts/filter/yodas2_sidon.py` joins consecutive captions into chunks, cuts those, runs the
cascade with CTC switched off, and only then calls Whisper on what survived.

**The gap rule is inert on this corpus.** Captions tile the video: over 2,207 boundaries the
median gap is 0.01 s and the maximum is 0.0 s, so any `--max-gap` above zero merges every
boundary. What actually decides where a chunk ends is `--target-duration`, which defaults to 8 s.
Keep the gap rule anyway — it is the only thing that would stop a chunk spanning a real silence in
a batch that has them.

Measured on the same 10 recordings (1.59 h) with and without joining:

| | cut on captions | joined at 8 s |
|---|---|---|
| chunks reaching the cascade | 2,217 | 603 |
| accepted | 218 (9.8%) | 313 (51.9%) |
| audio kept | 0.22 h | **0.81 h** |
| yield of source | 14% | **51%** |
| median clip | 3.47 s | 9.20 s |
| rejected `too_short` | 76.5% | 0.3% |
| transcripts with punctuation | 67.4% | 85.3% |
| ending in `.`/`?`/`!` | 60.1% | 70.0% |
| videos represented | 6 | 9 |
| Whisper calls | 432 (19%) | 502 (83%) |
| wall time | 3.8 min | 5.5 min |

**Joining is worth 3.7x the usable audio for 16% more Whisper calls.** The deferral saves much
less once the duration floor stops doing the rejecting (83% of chunks now reach Whisper against
19%), but the absolute transcription cost barely moves while the output triples.

Two things get worse and one does not improve:

- `multi_speaker` rises to 14.4% from 2.8%: a 9 s chunk has far more chance of catching a second
  voice than a 3 s caption. This is the stage doing its job, not a regression.
- `ctc_alignment` is 31.3% of all chunks against 9.7% before, but as a share of what Whisper
  actually saw it *fell*, 38% against 50%.
- **Chunks still start mid-sentence.** Only 21.7% begin with a capital, unchanged from 22.5%,
  because a joined chunk still begins on a caption boundary — there are simply fewer of them. The
  §7 route, which closes chunks on Whisper's own sentence punctuation, is the only one that fixed
  this (88.2%).

## 9. The source pass costs Polish half its yield

`scripts/filter/yodas2_sidon.py` now drops a whole recording when the diarisation stage rejected
`source_max_flag_rate` of the chunks it actually scored, the mechanic of
`scripts/filter/emilia.py` §4. The rate is taken over chunks that *reached* the stage — one
rejected on length was never diarised — and a recording under `source_min_clips` scored chunks is
left alone. Clips already written to disk are unlinked when their source is dropped.

Measured on the same 10 Polish recordings, the language with the worst per-clip rate (38.4%):

| threshold | recordings dropped | clips lost to it | clips kept | videos |
|---|---|---|---|---|
| off | — | — | 103 | 7 |
| 0.5 | 3 | 53 | 49 | 4 |
| 0.3 | 4 | 62 | 41 | 3 |

**At 0.5 the pass more than halves Polish's output**, and takes three of seven videos with it.
That is the stage working, not misfiring: a recording where half the diarised chunks carry a
second voice is an interview, and the chunks that happened to look clean are not trustworthy.

The default here is 0.5 rather than Emilia's 0.33, because the unit differs. An Emilia speaker is
one voice in one recording, so dropping the source costs one voice. A YODAS video is one recording
that may hold several, so dropping it discards its good speakers too — 0.3 cost 9 more clips than
0.5 for one more rejected recording.

## 10. Three guards against a second voice, and what each one catches

The corpus has no speaker field, so the pipeline defends against multi-speaker material three
times over. They are not redundant — each catches what the one before cannot.

| stage | question it asks | what it misses |
|---|---|---|
| `MultiSpeakerMetric` in the cascade | who speaks when, inside a window | two similar voices taking clean turns merge into one slot |
| source pass, §9 | what share of this recording's chunks were flagged | a good recording's one bad clip |
| `scripts/finalize/yodas2_sidon_prune.py` | does the voice itself change between two moments | overlapped speech, which embeds as neither voice |

The prune stage flags by duration-normalized z within a band rather than by raw similarity,
because similarity falls with clip length (`knowledge/emilia.md` §3).

Validated on a clip that reached a finalized subset despite holding two voices,
`YR8Wq-vTux4-00050-00022606-00023500`:

| | value |
|---|---|
| ECAPA similarity | 0.2867 |
| subset median | 0.4151 |
| duration-normalized z | −2.21 |
| rank | 3rd worst of 49 |

At the default `--max-z -2.0` it is flagged, together with 2 others, and removed from the wav,
the `.npy` and every CSV row. **It sits only 0.21 past the threshold**, so it is not a comfortable
catch — a subset drawn from cleaner recordings would shift the band mean and could let it through.

## 11. Speaker fixup matters more here than on Emilia, but needs a wide pool

`scripts/filter/yodas2_sidon_speaker_fixup.py` merges video IDs that are one person, the way
Emilia's does for diarisation labels. The case is stronger here: a channel host appears in every
video they publish, so one person routinely holds dozens of IDs.

It found **no merges across 7 videos** from 10 random recordings of one Polish shard, which is
the expected result rather than a failure — ten recordings drawn at random from a 219-recording
shard are unlikely to share a host. Run it over a whole shard or several, not over a sample.

## 12. Two failures that only appear on a CUDA box, and the silence behind one of them

### Forking after CUDA is initialised rejects everything, quietly

`scripts/filter/yodas2_sidon.py` loads Whisper onto the GPU in the parent *before* it starts the
worker pool. Linux's default start method is fork, and a forked child inherits a CUDA context it
cannot use, so every GPU metric in the worker raises `CUDA driver initialization failed`.

That is not a crash. `MultiSpeakerMetric.validate` catches, prints, and **returns False**, which
the cascade reads as a rejection — and since diarisation sits before NISQA, nothing downstream
ever runs. The symptom is a wall of identical messages and a run that rejects ~100% as
`multi_speaker`. The pool is now created with an explicit `spawn` context.

**macOS spawns by default, so this cannot be reproduced on a Mac.** Anything in this repo that
forks after touching the GPU has the same exposure; the other filters build only lazy Pipelines in
the parent, so they do not.

### 1.4% of chunks are digitally silent

Measured over 5,537 chunks across the Italian, Polish and Spanish batches: **77 (1.39%) are
exactly zero**, and none are merely near-silent. Caption spans do not always cover audio.

Silent audio used to fail `EffectiveBandwidthMetric` with a bare `StopIteration`, whose message is
the empty string — hence `EffectiveBandwidthMetric failed: ` with nothing after it, preceded by two
numpy warnings about an empty slice. The loud-frame filter keeps nothing when every frame carries
the same energy, so no band survives. The metric now says so, and still rejects.

### The VAD step

The cascade judges the audio that is there, not the audio that is missing, so a chunk that opens
or closes on dead air passes every stage. Silero VAD now runs on survivors, before Whisper is paid
for: ends longer than `--max-silence` are cut back to `--keep-silence`, and a gap that long
mid-clip is rejected as `SILENCE`, a new verdict. Trimming that leaves less than `min_duration`
also reports SILENCE rather than TOO_SHORT, because `reached()` treats TOO_SHORT as never having
been diarised and would undercount the source pass.

It fires rarely on this corpus — 1 rejection and 1 trim in 194 chunks of Italian — which follows
from captions tiling the video. It is there for the tail, not the median.

## 13. Caption spans overrun the audio, and the fix has a degenerate edge

Caption timestamps are not bounded by the file: on the Spanish batch the spans cover 101% of the
stated duration at the 90th percentile. Two failures follow, both of which killed long runs.

**A span starting past the end.** `SoundFile.seek()` answers a position beyond the file with
`Internal psf_fseek() failed` rather than a short read — unlike `read()`, which tolerates EOF. One
such span took down a run 9% in, after ~8,400 paid transcriptions. `chunk_spans` now takes the
real audio duration, drops a chunk starting at or past the end, and clamps one that merely ends
past it. `total_chunks` reads each audio header so the progress denominator applies the same rule.

**A clamped span rounding to nothing.** A chunk beginning a fraction of a second before the end
survives the clamp, and `int((end - begin) * rate)` then rounds to zero frames. `np.ptp` has no
identity on an empty array, so the silence check raised — and it sat *outside* `_score`'s try, so
the exception crossed the pool and ended a run at 76%, after ~83,000 transcriptions.

### The robustness posture that followed

A single unusable chunk out of 148,545 must not cost the hours behind it. So:

- `_score` does all its work inside the try, empty arrays included, and an empty chunk falls
  through to the cascade, which names it TOO_SHORT rather than SILENCE.
- `scored()` pulls pool results one at a time. `imap` re-raises a worker's exception at that
  item's position and later results still arrive, so a failed chunk is reported and the run
  continues. Verified against a task that cannot even be pickled.
- `utterances()` guards each seek separately from the file open, so one span a file cannot answer
  costs that span rather than the whole recording.

Every one of these reports the chunk through the filter's existing `failures` counter, so a run
that quietly starts failing everything is still visible in the summary.

## 14. What detects a second voice inside a clip, and what does not

Measured on Emilia EN (batches `en-b000000` and `en-b000007`) before that corpus was removed from the repo; kept
because the `speaker_drift` stage rests on it. Recorded so the failed approaches are not tried again.

| approach | result |
|---|---|
| Metadata (`dnsmos`, duration, text) | Nothing: the ASR transcribes both speakers inline |
| CTC `gap_speech` / `trail_speech` | Only catches an intruder the ASR dropped (a backchannel) |
| WavLM x-vector windows against a speaker anchor | **AUC 0.655 on unbiased clips** (0.828 on a score-selected set; the gap is selection bias) |
| Shorter windows for the above | Monotonically worse: 0.75 s 0.715, 1.0 s 0.730, 1.5 s 0.749, 2.0 s 0.801 |
| A classifier fitted on the two scores | Leave-one-out 0.766, worse than the unfitted product 0.818, on about 40 points |
| F0 shift across the clip | Disagreed with the listener on 4 of 12 clips: it tracks intonation, not identity |
| `pyannote/speaker-diarization-3.1`, full pipeline | 1 of 7 ear-confirmed clips: segmentation emits no second speaker for clustering to find |

**The anchorless, duration-normalised variant works at the tail.** `resonance/metrics/speaker_drift.py` scores the
minimum pairwise cosine between ECAPA windows of 2 s, with no speaker anchor, and the `speaker_drift` stage ranks it
by z within a duration band (`resonance/metrics/banding.py`). On 500 random Emilia clips that had already passed
the multi-speaker metric, the 10 lowest-z clips were labelled by ear: **7 confirmed multi-speaker** (p < 0.02 against
a 1/3 base rate).

Two properties carry it, and both were absent from the x-vector attempt:

- **No anchor** to average the intrusion away.
- **The duration control.** Raw similarity correlates -0.574 with clip length (more window pairs, lower minimum), so
  an unnormalised threshold is mostly a long-clip filter.

It is measured only as precision at the extreme tail. Recall is unknown. The score is unimodal, so there is no
valley to put a threshold in; z ≤ -2 is an outlier cut.

## 15. The refactored pipeline reproduces the old one

Measured on 2026-11-14 on a 2x RTX 4090 box (256 cores, torch 2.11.0+cu128), batch `en001/train-00091` (251
recordings, 37.1 h read) and a 40-recording subset of it. The old code is tag `pre-refactor`; the new code is
branch `refactor/pipeline`. Both were run on the same segments for the filter comparison. The comparison scripts
(`golden.sh`, `parity.py`, `compare.py`) lived on the box, outside the repo.

**The filters are identical clip for clip.**

| | 40 recordings | full batch |
|---|---|---|
| clips into the cascade | 811 | 5,948 |
| cascade verdicts identical, old vs new | 811 (100%) | 5,948 (100%) |
| clips kept after cascade, source pass and drift | 443 = 443, same clips | 3,221 = 3,221, same clips |
| largest drift-similarity difference | 5.0e-7 | 5.0e-7 |
| speaker IDs identical after fixup | 443 of 443 | 3,221 of 3,221 |

The old filter is itself deterministic: two runs of it gave identical verdicts on all 811 clips.

**Higgs codec tokens are identical**, on 40 of 40 clips of the 40-recording run (`finalize.codecs: [higgs]`
against the old `tools/codec/higgs.py`). The codec needs `pip install .[higgs]`, which adds `transformers` and
`accelerate`.

**The segmenter is identical when the thread caps match.** With `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` set to 4, the old segmenter produced exactly the new one's output on the
40 recordings: 811 of 811 clip names and transcripts. The old segmenter is also deterministic across runs and
across the two GPUs.

**The thread caps themselves change about 2% of the clips.** The new pipeline caps every worker's thread pools
(`resources.threads_per_worker`), which the old one did not. Old without caps against new with caps:

| | 40 recordings | full batch |
|---|---|---|
| clips, old / new | 811 / 811 | 5,940 / 5,948 |
| same recording, start and end | 809 | 5,900 (99.3%) |
| of those, same transcript | 800 | 5,797 (98.3%) |

Of the 103 differing transcripts on the full batch, 41 differ only in punctuation or case and 28 in one word. Neither
side is the correct one: thread count changes floating-point summation order in the feature computation, and beam
search amplifies it. The caps stay, because without them every worker starts one BLAS thread per core and a 256-core
box runs into its process limit.

### ⚑ Two settings the parity depends on

- **Quality workers set `torch.set_float32_matmul_precision("high")`** (`resonance/stages/quality.py`). `import
  speechbrain` switches TF32 matmuls on, and the old filter imported it in every worker for the drift metric, so
  every threshold in the config was calibrated with TF32. Without the explicit setting, 3 of 811 clips changed
  verdict, all on a CTC bound (for example `trail_speech` 0.0501 against a bound of 0.05).
- **Clip names carry a per-recording index**, so one changed boundary renames every later clip of that recording.
  Compare segmentations by recording, start and end, never by name: by name the full batch looked 90.6% identical
  when it was 99.3%.

## 16. Speed and disk of the refactored pipeline

Same box and date as §15. Config `configs/yodas2_sidon.yaml` defaults unless noted.

**One batch on one RTX 4090** (`en001/train-00091`, 5,948 clips segmented, 3,221 kept):

| stage | old | new |
|---|---|---|
| segment | 447 s | 418 s with 8 prep workers, 380 s with 24, **274 s with 24 and 3 Whisper loops** |
| filters | 168 s (cascade, source pass, drift) | 144 s quality + 20 s speaker_drift |
| speaker fixup | 19.5 s | 14 s |
| select, finalize (2,815 clips, deflate zip) | not part of the old driver | 61 s |

**Whisper is the bottleneck.** Of the 417 s of segmenting, Whisper takes 327 s. Read, resample and VAD run in the
prep workers; with 8 of them the GPU loop waited 60 s for audio, with 24 it waited 21 s. Both runs produced the same
5,948 clips and transcripts.

**A larger Whisper batch does not help.** One GPU, 24 prep workers:

| `segment.batch_size` | Whisper | whole stage | clips |
|---|---|---|---|
| **16 (default)** | **328 s** | **379 s** | 5,948 |
| 32 | 330 s | 382 s | 5,925 |
| 64 | 334 s | 389 s | 5,926 |

The time does not move, and the output does: a different batch changes which windows are decoded together. The
default stays at 16.

**Several Whisper loops per GPU do help, with identical output.** One loop leaves the GPU at 63%: the Python loop
around Whisper is the limit, not the GPU. `segment.workers` runs that many loops on one GPU, each a process with its
own model and its own share of the batch's recordings. Same batch, 24 prep workers in total:

| `segment.workers` | segment stage | GPU utilisation | GPU memory |
|---|---|---|---|
| 1 | 390 s | 63% | 4.6 GB |
| 2 | 295 s | 82% | 9.2 GB |
| **3 (default)** | **274 s (-30%)** | 92% | 13.1 GB |
| 4 | 263 s | 97% | 18.4 GB |
| 6 | CUDA out of memory on 24 GB | | |

`segment.csv` is byte-identical for 1, 2, 3 and 4 loops (5,948 clips). The default is 3: 4 is 4% faster and leaves
about 5 GB free on a 24 GB card.

**Confirmed end to end:** the same four batches (`00092` to `00095`) on both GPUs with `segment.workers: 3` and
`cpu_workers: 24` took 17.3 min for the batch stages, 4.3 min per batch, against 21.5 min before, with the same
14,799 kept and 10,917 selected clips. With both GPUs busy a batch segments in 4.6-5.5 min, slower than the 274 s
measured on one GPU alone.

⚑ When a loop fails, its helper processes must die with it. The 6-loop run first hung forever: the other loops were
terminated, their prep workers were orphaned, and Python's exit waited on them. `gpu.map_gpus` now gives each shard
its own process group and kills the group; the same run then failed in 32 s with the out-of-memory error.

**Four batches on both GPUs, download to `audio.zip`** (`en001/train-00092` to `00095`, 16 prep workers per GPU):

| | |
|---|---|
| wall time, whole run | **25.8 min** (1,548 s) |
| batch stages for the 4 batches | 21.5 min, so 5.4 min per batch (old: about 10.3 min per batch on one GPU) |
| per batch on its GPU | segment 6.7-7.4 min, quality 2.4-2.8 min, speaker_drift 20-24 s |
| clips | 27,455 segmented (72.6 h), 14,799 kept (38.5 h, 54%), 10,917 selected (28.6 h) |
| speakers | 509 video IDs, 473 after fixup (largest cluster 8), 406 after select |
| speaker_fixup, select, finalize | 18 s, under 1 s, 3 min 51 s |
| errors | 0 in every stage |

Per GPU the throughput is what it was. The gain is that both GPUs are used without splitting the batches by hand.

**Disk.**

- `discard_rejected` freed 1.2-1.7 GB per batch (2,700-4,000 wavs).
- After the run, `clips/` held 1.7 GB (the 3,882 kept clips that select left out) and the dataset 3.75 GB.
- **Deflate saves about 19%:** the 10,917 selected wavs are 4.60 GB and `audio.zip` is 3.74 GB.

**`scripts/setup.sh` on a clean HF cache** took 72 s: pip install, six model downloads (1.7 GB in the HF cache) and
the preflight checks.

## 17. Linking the same person across videos

Measured on Emilia EN (634 speakers with at least 4 clips, 200,661 pairs) before that corpus was removed; kept
because `speaker_fixup` rests on it.

- **Use `pyannote/wespeaker-voxceleb-resnet34-LM`**: 256-d, about 0.017 s per clip, whole clips truncated to 8 s.
  Within-speaker cosine (same source, split halves) has median 0.869 and p5 0.728; between-speaker has median 0.092
  and p99 0.835.
- **Do not use `microsoft/wavlm-base-plus-sv` for this.** Its between-speaker median is 0.71 with p99 0.984, so it
  links 3% of all pairs at 0.95 or above.
- **Compare sources over clip pairs, not centroids.** A centroid's cosine ceiling moves with the clip count: over 15
  unrelated sources it climbs 0.735 (1 clip), 0.869 (4), 0.917 (16), and overtakes true matches from 3 clips up, so a
  threshold calibrated at one run size is wrong at the next. The share of clip pairs above cosine 0.75 holds still:
  60-64 merges whether 1 or 4 clips are sampled. This is what `resonance/speakers/clustering.py` does.
- **A clip pair at 0.95 or above is duplicated audio**, not a speaker link. `merges.csv` flags it.
- **Run it over many batches.** Ten random recordings rarely share a host (§11); four batches merged 36 of 509
  video IDs (§16).

