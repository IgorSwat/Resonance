import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from tools.load.constants import AUDIO_EXTENSIONS, AUDIO_FIELD, DEFAULT_BUFFER_SIZE


class ParquetAudioLoader:
    """
    Yields decoded audio rows from a parquet file, at most buffer_size held in memory.

    The column holding audio files is detected (or given as audio_column) and exposed as 'audio';
    remaining columns are passed through unchanged.
    """

    def __init__(self, path, buffer_size=DEFAULT_BUFFER_SIZE, audio_column=None):
        self.file = pq.ParquetFile(path)
        self.buffer_size = buffer_size
        self.audio_column = audio_column or self._detect_audio_column()

    def _detect_audio_column(self):
        for batch in self.file.iter_batches(batch_size=1):
            for name, value in zip(batch.schema.names, batch.to_pylist()[0].values()):
                if self._extension(value) in AUDIO_EXTENSIONS:
                    return name
            break

        raise ValueError("No audio column found")

    def __iter__(self):
        for batch in self.file.iter_batches(batch_size=self.buffer_size):
            for row in batch.to_pylist():
                value = row.pop(self.audio_column)
                audio, sample_rate = sf.read(self._source(value), dtype="float32")
                yield {AUDIO_FIELD: audio, "sample_rate": sample_rate, **row}

    @staticmethod
    def _extension(value):
        if isinstance(value, dict):
            value = value.get("path")

        return Path(value).suffix.lower() if isinstance(value, str) else ""

    @staticmethod
    def _source(value):
        """
        A row's audio is either a file path or embedded bytes (optionally in a {path, bytes} struct).
        """

        if isinstance(value, dict):
            return io.BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
        return io.BytesIO(value) if isinstance(value, bytes) else value
