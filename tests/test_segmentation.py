from resonance.config.schema import SegmentConfig
from resonance.segmentation import packing, sentences, transcribe
from resonance.segmentation.abbreviations import ABBREVIATIONS
from resonance.segmentation.sentences import Word


def words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(start + index * step, start + index * step + 0.4, f" {token}")
            for index, token in enumerate(text.split())]


def test_terminal_respects_abbreviations_and_ordinals():
    english = ABBREVIATIONS["en"]
    assert sentences.terminal(" done.", english) and sentences.terminal(' right?"', english)
    assert not sentences.terminal(" Dr.", english) and not sentences.terminal(" and", english)
    assert not sentences.terminal(" 3.", set(), ordinal_dot=True) and sentences.terminal(" 3.", set())


def test_split_closes_on_marks_and_opens_on_pauses():
    spoken = words("Hello there. This is cut") + words("off. Next one.", start=10.0)
    found = sentences.split(spoken, max_pause=2.0, abbreviations=set())
    assert [(len(s.words), s.closed) for s in found] == [(2, True), (3, False), (1, True), (2, True)]


def test_windows_group_and_split():
    assert transcribe.windows([(0, 10), (12, 25), (40, 45)]) == [{"start": 0, "end": 25}, {"start": 40, "end": 45}]
    assert transcribe.windows([(0, 70)]) == [{"start": 0, "end": 30.0}, {"start": 30.0, "end": 60.0},
                                             {"start": 60.0, "end": 70}]
    assert transcribe.foreign_overlap(5, 8, [(7, 9)]) and not transcribe.foreign_overlap(5, 7, [(7, 9)])


def test_gaps_and_boundary():
    silences = packing.gaps([(1.0, 4.0), (4.1, 6.0), (7.0, 9.0)], duration=10.0, min_gap=0.15)
    assert silences == [(0.0, 1.0), (6.0, 7.0), (9.0, 10.0)]
    assert packing.boundary(silences, before=5.9, after=7.1) == (6.0, 7.0)
    assert packing.boundary(silences, before=2.0, after=3.0) is None


def test_pack_cuts_in_pauses_with_lead_and_trail():
    config = SegmentConfig(target_duration=4.0, min_duration=3.0, max_duration=25.0)
    # speech 1.0-5.4 and 7.0-9.9, silence between; two closed sentences in the first stretch, one in the second
    first = sentences.split(words("One two three four. Five six seven eight nine.", start=1.0), 2.0, set())
    second = sentences.split(words("Ten eleven twelve thirteen fourteen fifteen.", start=7.0), 2.0, set())
    silences = packing.gaps([(1.0, 5.4), (7.0, 9.9)], duration=12.0, min_gap=0.15)
    utterances = packing.pack(first + second, silences, config)
    assert [text for _, _, text in utterances] == ["One two three four. Five six seven eight nine.",
                                                   "Ten eleven twelve thirteen fourteen fifteen."]
    begin, end, _ = utterances[0]
    assert begin == 1.0 - config.lead and end == 5.4 + config.trail


def test_pack_skips_open_and_lowercase_sentences():
    config = SegmentConfig(target_duration=2.0, min_duration=1.0)
    silences = [(0.0, 1.0), (4.0, 5.0), (8.0, 9.0)]
    lowercase = sentences.split(words("and so it ends here now.", start=1.0), 2.0, set())
    unclosed = sentences.split(words("This never ends", start=5.0), 2.0, set())
    assert packing.pack(lowercase + unclosed, silences, config) == []
