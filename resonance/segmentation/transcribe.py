"""Whisper over a recording's speech: every word with its timestamps, and the windows in another language."""

from typing import Any

import numpy as np

from resonance.segmentation.sentences import Word

WINDOW = 30.0                   # Whisper's input length; VAD segments are grouped up to it
NUMERALS = "0123456789%$£€"     # a token holding any of these is one Whisper may not write when spelling numbers

Span = tuple[float, float]


def load(model: str, spell_numbers: bool = False) -> Any:
    """
    faster-whisper's batched pipeline, also recording the language Whisper detects in each window.

    Decoding stays forced to the corpus language, so a foreign window comes out as a *translation* rather than as
    its own language, and nothing in the text gives it away. The language is read off the encoder output the batch
    has already computed, which costs one decoder step per window.
    """

    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    class Pipeline(BatchedInferencePipeline):
        def generate_segment_batched(self, features: Any, tokenizer: Any, options: Any) -> tuple[Any, Any]:
            encoder_output, output = super().generate_segment_batched(features, tokenizer, options)
            self.languages.extend(probabilities[0][0].strip("<|>")
                                  for probabilities in self.model.model.detect_language(encoder_output))
            return encoder_output, output

    cuda = torch.cuda.is_available()
    pipeline = Pipeline(WhisperModel(model, device="cuda" if cuda else "cpu",
                                     compute_type="float16" if cuda else "int8"))
    # Without its digit and currency tokens Whisper has to write a number the way it was said ("twenty twenty"),
    # so the reading comes from the audio instead of from a verbalizer's guess. -1 keeps Whisper's own defaults.
    tokenizer = pipeline.model.hf_tokenizer
    pipeline.suppressed = [-1] + ([token for token in range(tokenizer.get_vocab_size())
                                  if any(character in NUMERALS for character in tokenizer.decode([token]))]
                                 if spell_numbers else [])
    return pipeline


def windows(speech: list[Span]) -> list[dict[str, float]]:
    """VAD segments grouped into Whisper windows of at most WINDOW seconds, in seconds."""

    grouped: list[dict[str, float]] = []
    for start, end in speech:
        while end - start > WINDOW:     # one segment longer than a window is split at WINDOW
            grouped.append({"start": start, "end": start + WINDOW})
            start += WINDOW
        if grouped and end - grouped[-1]["start"] <= WINDOW:
            grouped[-1]["end"] = end
        else:
            grouped.append({"start": start, "end": end})
    return grouped


def transcribe(pipeline: Any, y: np.ndarray, speech: list[Span], language: str, batch_size: int,
               beam_size: int) -> tuple[list[Word], list[Span]]:
    """
    Every word Whisper hears in the span, with its timestamps in span seconds, and the windows whose language
    Whisper detects as something other than `language`, as (start, end) in span seconds.
    """

    grouped = windows(speech)
    pipeline.languages = []
    segments, _ = pipeline.transcribe(y, language=language, batch_size=batch_size, beam_size=beam_size,
                                      word_timestamps=True, clip_timestamps=grouped,
                                      suppress_tokens=pipeline.suppressed)
    # kept as Whisper spells it, leading space included: a word fragment such as "-term" or ".S."
    # comes without one, and joining stripped tokens with spaces wrote "long -term" and "U .S."
    words = [Word(w.start, w.end, w.word)
             for segment in segments for w in (segment.words or []) if w.word.strip()]
    if len(pipeline.languages) != len(grouped):
        raise RuntimeError(f"{len(pipeline.languages)} language detections for {len(grouped)} windows")
    foreign = [(window["start"], window["end"]) for window, detected in zip(grouped, pipeline.languages)
               if detected != language]
    return words, foreign


def spell(pipeline: Any, y: np.ndarray, language: str, beam_size: int) -> str:
    """One clip's transcript from a pipeline loaded with spell_numbers: every number written as it was said."""

    segments, _ = pipeline.model.transcribe(y, language=language, beam_size=beam_size, without_timestamps=True,
                                            condition_on_previous_text=False, suppress_tokens=pipeline.suppressed)
    return "".join(segment.text for segment in segments).strip()


def foreign_overlap(begin: float, end: float, foreign: list[Span]) -> bool:
    return any(start < end and begin < stop for start, stop in foreign)
