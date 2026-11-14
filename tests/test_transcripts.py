from pathlib import Path

from resonance.config.schema import TranscriptFixupConfig
from resonance.io import manifest
from resonance.stages.transcript_fixup import decide
from resonance.transcripts.normalize import Normalizer, has_numbers


def row(name: str, text: str) -> dict[str, str]:
    return {"dataset": "d", "name": name, "transcription": text, "speaker_id": "s", "language": "en",
            "duration": "5.0", "audio_path": f"clips/{name}.wav", "source_id": "s"}


def test_has_numbers() -> None:
    assert has_numbers("in 1994") and has_numbers("costs $five") and has_numbers("ten%") and not has_numbers("ten")


def test_confirmed_compares_only_the_numbers_in_order() -> None:
    normalizer = Normalizer("en", TranscriptFixupConfig())
    text = "On this date, 1956, about 35 senators of the U.S. House."
    assert normalizer.confirmed(text, "uh on this date in Nineteen Fifty Six about thirty-five senators of the US house")
    assert not normalizer.confirmed(text, "On this date, about thirty five senators.")              # a number is missing
    assert not normalizer.confirmed(text, "thirty five senators on this date nineteen fifty six")    # out of order
    assert not normalizer.confirmed("I'm like 145.", "I'm like one forty five.")                     # another reading


def test_normalizer_roles() -> None:
    normalizer = Normalizer("en", TranscriptFixupConfig())
    assert normalizer.reading("Dr. Smith paid $5 & left.") == "Dr. Smith paid five dollars & left."
    assert normalizer.normalized("Dr. Smith  paid $5 & left .") == "Doctor Smith paid five dollars and left."
    plain = Normalizer("en", TranscriptFixupConfig(terms=False))
    assert plain.normalized("Dr. Smith paid $5.") == "Dr. Smith paid five dollars."


def test_decide_keeps_confirmed_numbers_and_drops_the_rest() -> None:
    rows = [row("plain", "No numbers here, Dr. Lee."), row("agree", "It opened in 1994."),
            row("missing", "It closes on November 30, so hurry."), row("other", "I'm like 145.")]
    spelled = {"agree": "Uh, it opened in nineteen ninety-four.", "missing": "It closes on November, so hurry.",
               "other": "I'm like one forty five."}
    kept, dropped = decide(rows, spelled, Normalizer("en", TranscriptFixupConfig()))
    assert [r["name"] for r in kept] == ["plain", "agree"] and [r["name"] for r in dropped] == ["missing", "other"]
    assert kept[0][manifest.NORMALIZED] == "No numbers here, Doctor Lee."
    assert kept[1][manifest.NORMALIZED] == "It opened in nineteen ninety-four."
    assert kept[1]["transcription"] == "It opened in 1994."             # the original column is untouched
    assert dropped[1]["whisper"] == "I'm like one forty five." and "one hundred" in dropped[1]["verbalis"]


def test_normalized_column_travels_and_sits_beside_the_transcription(tmp_path: Path) -> None:
    rows = [{**row("a", "In 1994."), manifest.NORMALIZED: "In nineteen ninety-four."}]
    manifest.write(tmp_path / "pool.csv", rows)
    assert manifest.read(tmp_path / "pool.csv") == rows
    manifest.write(tmp_path / "final.csv", rows, manifest.columns_of(rows, manifest.STANDARD))
    assert list(manifest.read(tmp_path / "final.csv")[0]) == [
        "dataset", "name", "transcription", "transcription_normalized", "speaker_id", "language"]
    manifest.write(tmp_path / "old.csv", [row("a", "x")], manifest.columns_of([row("a", "x")], manifest.STANDARD))
    assert list(manifest.read(tmp_path / "old.csv")[0]) == list(manifest.STANDARD)
