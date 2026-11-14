from resonance.io import manifest


def test_roundtrip_keeps_awkward_text(tmp_path):
    rows = [{"dataset": "d", "name": "a", "transcription": "pipe | here, \"quotes\" and a bare \r return\\",
             "speaker_id": "s", "language": "en", "duration": "3.5", "audio_path": "clips/a.wav", "source_id": "s"},
            {"dataset": "d", "name": "b", "transcription": "Plain.", "speaker_id": "s", "language": "en",
             "duration": "1.5", "audio_path": "clips/b.wav", "source_id": "s"}]
    path = tmp_path / "m.csv"
    manifest.write(path, rows)
    assert manifest.read(path) == rows
    assert path.read_bytes().count(b"\r\n") == 3 and not (tmp_path / "m.csv.tmp").exists()
    assert manifest.hours(rows) == 5 / 3600


def test_standard_columns_only(tmp_path):
    row = dict.fromkeys(manifest.COLUMNS, "x")
    manifest.write(tmp_path / "m.csv", [row], manifest.STANDARD)
    assert list(manifest.read(tmp_path / "m.csv")[0]) == list(manifest.STANDARD)
