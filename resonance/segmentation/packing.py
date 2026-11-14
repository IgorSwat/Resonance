"""Whole sentences packed into utterances that start and end in real pauses."""

from resonance.config.schema import SegmentConfig
from resonance.segmentation.sentences import Sentence

TOLERANCE = 0.3                 # how far a word timestamp may sit from the VAD boundary it matches
OPENING = "¿¡«\"“'(["           # leading characters that may precede a sentence's first letter

Span = tuple[float, float]


def gaps(speech: list[Span], duration: float, min_gap: float) -> list[Span]:
    """The VAD silences of at least min_gap, including the span's two ends, as (start, end)."""

    edges = [0.0] + [t for segment in speech for t in segment] + [duration]
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_gap]


def boundary(silences: list[Span], before: float, after: float) -> Span | None:
    """
    The longest silence between two words (before: the earlier word's end, after: the later's start), allowing
    TOLERANCE for Whisper's timing; None when they are not separated by a pause.
    """

    candidates = [(a, b) for a, b in silences if a < after + TOLERANCE and b > before - TOLERANCE]
    return max(candidates, key=lambda gap: gap[1] - gap[0], default=None)


def pack(sentences: list[Sentence], silences: list[Span], config: SegmentConfig) -> list[tuple[float, float, str]]:
    """
    Whole closed sentences packed into utterances of about target_duration, cut in real pauses with `lead`
    seconds of silence before the speech and `trail` after, as far as the pause allows; (begin, end, text).

    A sentence Whisper did not close with punctuation, or one interrupted by a long pause, is never part of an
    utterance: it is where speech was cut, overlapped or abandoned.
    """

    utterances = []
    index = 0
    while index < len(sentences):
        first = sentences[index]
        previous_end = sentences[index - 1].end if index else 0.0
        opening = boundary(silences, previous_end, first.start)
        # a sentence resumed after a long pause is not capitalised by Whisper; one that is starts
        # a Spanish question or exclamation opens on ¿/¡, a quote on its mark: the letter after decides
        initial = first.words[0].text.strip().lstrip(OPENING)[:1]
        if not first.closed or opening is None or not initial.isupper() and not initial.isdigit():
            index += 1
            continue

        # extend sentence by sentence; remember the last place the utterance could be closed
        best, last = None, index
        while last < len(sentences) and sentences[last].closed:
            sentence = sentences[last]
            if last > index and sentence.start - sentences[last - 1].end > config.max_pause:
                break
            if sentence.end - first.start > config.max_duration:
                break
            following = sentences[last + 1].start if last + 1 < len(sentences) else sentence.end + 1.0
            closing = boundary(silences, sentence.end, following)
            if closing is not None:
                length = sentence.end - first.start
                if length >= config.target_duration:
                    # close here or at the previous clean pause, whichever lands nearer the target
                    earlier = best and best[2] >= config.min_duration and \
                        config.target_duration - best[2] < length - config.target_duration
                    best = best if earlier else (last, closing, length)
                    break
                best = (last, closing, length)
            last += 1

        if best is None:
            index += 1
            continue
        last, closing, _ = best
        speech_start, speech_end = opening[1], closing[0]
        begin = speech_start - min(config.lead, opening[1] - opening[0])
        end = speech_end + min(config.trail, closing[1] - closing[0])
        if config.min_duration <= end - begin <= config.max_duration:
            text = "".join(word.text for sentence in sentences[index:last + 1]
                           for word in sentence.words).strip()
            utterances.append((begin, end, text))
        index = last + 1
    return utterances
