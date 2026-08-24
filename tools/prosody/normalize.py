import numpy as np

from tools.prosody.constants import (
    ADAPTED_CEILING,
    ADAPTED_CEILING_RATIO,
    ADAPTED_FLOOR,
    ADAPTED_FLOOR_RATIO,
    MIN_VOICED_FRAMES,
    OCTAVE_GUARD,
)


def clip_median(f0):
    """Pass 1: a clip's median voiced F0 in Hz, or None if it carries too little voicing."""

    voiced = ~np.isnan(f0)
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None
    return float(np.median(f0[voiced]))


def speaker_reference(clip_medians):
    """
    The speaker's habitual pitch, pooled over all of their clips.

    Median rather than mean, so octave-doubled clips cannot drag it up. Pooling is what makes
    the features behavioural: a per-clip reference would set every clip's mean to zero and
    erase the difference between a speaker's animated and flat recordings.
    """

    return float(np.median(np.asarray(clip_medians, dtype=float)))


def adapted_range(reference_hz):
    """Pass 2 search range, tied to the speaker — the main defence against octave errors."""

    return (
        max(ADAPTED_FLOOR, ADAPTED_FLOOR_RATIO * reference_hz),
        min(ADAPTED_CEILING, ADAPTED_CEILING_RATIO * reference_hz),
    )


def within_octave_guard(median_hz, reference_hz):
    """Whether a clip sits close enough to its speaker's reference for its contour to be trusted."""

    return OCTAVE_GUARD[0] * reference_hz <= median_hz <= OCTAVE_GUARD[1] * reference_hz


def to_semitones(f0, reference_hz):
    """
    Speaker-relative log pitch: 12 * log2(f0 / reference).

    Pitch is heard multiplicatively, so measured in Hz a high voice looks more variable than a
    low one purely as an artifact. Dividing by the speaker's own reference first makes the
    number mean "how far did this person move, for them" — behaviour rather than anatomy.
    """

    return 12 * np.log2(f0 / reference_hz)
