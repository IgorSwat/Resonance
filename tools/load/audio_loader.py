from pathlib import Path

import soundfile as sf

from tools.load.constants import AUDIO_EXTENSIONS, AUDIO_FIELD, DEFAULT_BUFFER_SIZE


class AudioLoader:
    """
    Yields decoded audio files (via soundfile) from a directory, at most buffer_size held in memory.
    """

    def __init__(self, directory, buffer_size=DEFAULT_BUFFER_SIZE):
        self.directory = Path(directory)
        self.buffer_size = buffer_size

    def paths(self):
        return sorted(
            path
            for path in self.directory.iterdir()
            if path.suffix.lower() in AUDIO_EXTENSIONS
        )

    def __iter__(self):
        paths = self.paths()
        for start in range(0, len(paths), self.buffer_size):
            buffer = [self._load(path) for path in paths[start : start + self.buffer_size]]
            yield from buffer

    @staticmethod
    def _load(path):
        audio, sample_rate = sf.read(path, dtype="float32")
        return {AUDIO_FIELD: audio, "sample_rate": sample_rate, "path": str(path)}
