"""The stages a config's `pipeline` list may name. A new module is one file in resonance/stages/ and one line here."""

import importlib

from resonance.pipeline.stage import Stage

STAGES = {
    "fetch": "resonance.stages.fetch:Fetch",
    "segment": "resonance.stages.segment:Segment",
    "quality": "resonance.stages.quality:Quality",
    "source_pass": "resonance.stages.source_pass:SourcePass",
    "speaker_drift": "resonance.stages.speaker_drift:SpeakerDrift",
    "discard_rejected": "resonance.stages.discard_rejected:DiscardRejected",
    "merge": "resonance.stages.merge:Merge",
    "transcript_fixup": "resonance.stages.transcript_fixup:TranscriptFixup",
    "speaker_fixup": "resonance.stages.speaker_fixup:SpeakerFixup",
    "select": "resonance.stages.select:Select",
    "highlights": "resonance.stages.highlights:Highlights",
    "finalize": "resonance.stages.finalize:Finalize",
}


def get(name: str) -> type[Stage]:
    if name not in STAGES:
        raise KeyError(f"Unknown stage {name!r}; known: {', '.join(STAGES)}")
    module, _, attribute = STAGES[name].partition(":")
    return getattr(importlib.import_module(module), attribute)
