"""
A language's batches cut into chunks, one pipeline run each, named the way the Drive folders are.

A chunk covers the batches [k * size, (k + 1) * size) of one shard, the last one of a shard shorter, and is named
<lang><shard>_<last + 1>: en004_50 holds en004/train-00000..49, en002_132 holds en002/train-00100..131.
"""

from dataclasses import dataclass, field

from resonance.config.schema import BatchRange
from resonance.datasets import yodas2_sidon
from resonance.pipeline.artifacts import BatchRef


@dataclass
class Chunk:
    language: str
    shard: str
    first: int
    last: int
    skipped: list[int] = field(default_factory=list)     # quarantined batches, left out of the run

    @property
    def name(self) -> str:
        return f"{self.language}{self.shard}_{self.last + 1}"

    def ranges(self) -> list[BatchRange]:
        """The chunk's batches as source.batches ranges, around the skipped ones."""

        result: list[BatchRange] = []
        for index in range(self.first, self.last + 1):
            if index in self.skipped:
                continue
            if result and result[-1].last == index - 1:
                result[-1].last = index
            else:
                result.append(BatchRange(self.shard, index, index))
        return result


def plan(language: str, size: int, shards: list[str] | None = None, start: BatchRef | None = None) -> list[Chunk]:
    """
    Every chunk of the language in order, from `start` on. `shards` limits it to those shard numbers ("004").

    A start inside a chunk makes that chunk begin at the start; it keeps the chunk's name, since the name only
    records where a chunk ends.
    """

    if start is not None and start.language != language:
        raise ValueError(f"--start {start.key} is not a {language} batch")
    chunks = []
    for shard_name in yodas2_sidon.shards(language):
        shard = shard_name[len(language):]
        if shards and shard not in shards or start is not None and shard < start.shard:
            continue
        count = len(yodas2_sidon.available(shard_name))
        for first in range(0, count, size):
            chunk = Chunk(language, shard, first, min(first + size, count) - 1)
            if start is not None and shard == start.shard:
                if chunk.last < start.index:
                    continue
                chunk.first = max(chunk.first, start.index)
            chunks.append(chunk)
    return chunks
