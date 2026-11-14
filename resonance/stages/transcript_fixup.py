"""
transcript_fixup: ClipManifest -> ClipManifest. Adds a normalized transcript, and drops clips whose numbers
cannot be confirmed.

Every clip gets a transcription_normalized column: its transcript run through Verbalis, with numbers, symbols and
abbreviations written as words. A clip whose transcript holds a number is first checked: Whisper transcribes it
again with its digit tokens suppressed, so it has to write the number the way it was said, and the clip stays only
if that spelling contains every number reading Verbalis gives, in order. The transcription column is left as it
is. transcript_fixup_dropped.csv lists every dropped clip with both readings.
"""

from pathlib import Path
from typing import Callable

from resonance.io import manifest
from resonance.io.manifest import Row
from resonance.pipeline import gpu
from resonance.pipeline.artifacts import ClipManifest
from resonance.pipeline.stage import Context, Stage
from resonance.transcripts.normalize import Normalizer, has_numbers

DROPPED_COLUMNS = ("name", "transcription", "verbalis", "whisper")


def spell_clips(task: tuple[str, str, int, list[tuple[str, Path]]], advance: Callable[[int], None]) -> dict[str, str]:
    """Each clip's transcript with its numbers spelled out by Whisper, as {name: text}; one process on one GPU."""

    from resonance.io.audio import read_audio, resample
    from resonance.segmentation import transcribe, vad

    model, language, beam_size, clips = task
    pipeline = transcribe.load(model, spell_numbers=True)
    spelled = {}
    for name, path in clips:
        audio = read_audio(path)
        spelled[name] = transcribe.spell(pipeline, resample(audio["audio"], audio["sample_rate"], vad.MODEL_RATE),
                                         language, beam_size)
        advance(1)
    return spelled


def decide(rows: list[Row], spelled: dict[str, str], normalizer: Normalizer) -> tuple[list[Row], list[Row]]:
    """
    (kept, dropped). A kept row gains its normalized transcript. A row with a number is dropped when Whisper's
    spelling lacks one of Verbalis's number readings, or when a digit survives normalization.
    """

    kept, dropped = [], []
    for row in rows:
        text = row["transcription"]
        if has_numbers(text):
            if not normalizer.confirmed(text, spelled[row["name"]]) or has_numbers(normalizer.normalized(text)):
                dropped.append({"name": row["name"], "transcription": text, "verbalis": normalizer.reading(text),
                                "whisper": spelled[row["name"]]})
                continue
        kept.append({**row, manifest.NORMALIZED: normalizer.normalized(text)})
    return kept, dropped


class TranscriptFixup(Stage):
    name = "transcript_fixup"
    scope = "pool"
    consumes = ClipManifest
    produces = ClipManifest
    needs_gpu = True

    def output(self, ctx: Context) -> ClipManifest:
        return ClipManifest(ctx.work_dir / "transcript_fixup.csv")

    def run(self, source: ClipManifest, ctx: Context) -> ClipManifest:
        config, language = self.config.transcript_fixup, self.config.source.language
        rows = manifest.read(source.path)
        numbered = [(row["name"], ctx.layout.root / row["audio_path"]) for row in rows
                    if has_numbers(row["transcription"])]
        spelled: dict[str, str] = {}
        if numbered:
            shares = gpu.split(numbered, len(ctx.gpus) * config.workers)
            for part in gpu.map_gpus(spell_clips, [(config.whisper, language, config.beam_size, share)
                                                   for share in shares], ctx.gpus,
                                     self.config.resources.threads_per_worker, ctx.log.progress(len(numbered)),
                                     self.log_path(ctx)):
                spelled.update(part)

        kept, dropped = decide(rows, spelled, Normalizer(language, config))
        manifest.write(ctx.work_dir / "transcript_fixup_dropped.csv", dropped, DROPPED_COLUMNS)
        manifest.write(ctx.work_dir / "transcript_fixup.csv", kept)
        ctx.summary.update(clips_in=len(rows), clips_out=len(kept), hours_out=round(manifest.hours(kept), 3),
                           with_numbers=len(numbered), confirmed=len(numbered) - len(dropped))
        return self.output(ctx)
