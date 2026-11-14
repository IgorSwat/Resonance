"""Whisper's words grouped into sentences."""

import dataclasses

TERMINAL = (".", "?", "!")
CLOSING = "\"'”’)]"             # trailing characters that may follow a sentence's final mark


@dataclasses.dataclass
class Word:
    start: float
    end: float
    text: str                   # as Whisper spells it, leading space included


@dataclasses.dataclass
class Sentence:
    words: list[Word]
    closed: bool                # ends on a terminal mark rather than on a long pause or the span end

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def split(words: list[Word], max_pause: float, abbreviations: set[str], ordinal_dot: bool = False) -> list[Sentence]:
    """Sentences, closed on a terminal mark; a pause over max_pause leaves the sentence open."""

    sentences, current = [], []
    for word in words:
        if current and word.start - current[-1].end > max_pause:
            sentences.append(Sentence(current, closed=False))
            current = []
        current.append(word)
        if terminal(word.text, abbreviations, ordinal_dot):
            sentences.append(Sentence(current, closed=True))
            current = []
    if current:
        sentences.append(Sentence(current, closed=False))
    return sentences


def terminal(text: str, abbreviations: set[str], ordinal_dot: bool = False) -> bool:
    """Whether a word ends a sentence: a final . ? ! that is not a known abbreviation (nor, with ordinal_dot, a
    number with a dot)."""

    stripped = text.strip().rstrip(CLOSING)
    if ordinal_dot and stripped.endswith(".") and stripped[:-1].isdigit():
        return False
    return stripped.endswith(TERMINAL) and stripped.lower() not in abbreviations
