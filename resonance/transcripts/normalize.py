"""
Transcripts as they are read aloud, with Verbalis.

Verbalis reads a number from the text alone, so its reading is a guess: "1800" may have been said "eighteen
hundred". It is trusted only where Whisper, made to spell numbers out from the audio, wrote the same reading.
"""

import re

from verbalis import Options, Verbalizer

from resonance.config.schema import TranscriptFixupConfig
from resonance.segmentation.transcribe import NUMERALS

_NUMERAL = re.compile(f"[{re.escape(NUMERALS)}]")


class Normalizer:
    """Verbalis in its two roles: reading numbers for the check, and writing the normalized transcript."""

    def __init__(self, language: str, config: TranscriptFixupConfig) -> None:
        self.numbers = Verbalizer(language, Options(normalize=False, trim=False, terms=False, symbols=False))
        self.full = Verbalizer(language, Options(normalize=config.normalize, trim=config.trim, terms=config.terms))

    def reading(self, text: str) -> str:
        """The text with only its numbers verbalized: what the spelled transcript is compared with."""

        return self.numbers.process(text)

    def confirmed(self, text: str, spelled: str) -> bool:
        """
        Whether every number reading Verbalis gives for `text` occurs in `spelled`, in the same order.

        Only the numbers are compared. Transcribing a clip a second time, Whisper writes small things differently
        (an "uh", "US" for "U.S."), and that says nothing about how a number was read.
        """

        heard, position = _words(spelled), 0
        for change in self.numbers.process_with_changes(text).changes:
            wanted = _words(change.replacement)
            if not wanted:
                continue
            position = next((start + len(wanted) for start in range(position, len(heard) - len(wanted) + 1)
                             if heard[start:start + len(wanted)] == wanted), None)
            if position is None:
                return False
        return True

    def normalized(self, text: str) -> str:
        return self.full.process(text)


def has_numbers(text: str) -> bool:
    return bool(_NUMERAL.search(text))


def _words(text: str) -> list[str]:
    return "".join(character if character.isalpha() or character == "'" else " "
                   for character in text.lower().replace("’", "'")).split()
