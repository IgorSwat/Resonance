# YODAS2-Sidon — what the Spanish batch actually contains

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
