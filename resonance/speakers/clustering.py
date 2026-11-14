"""
Which sources are one person: clip-pair agreement between sources, then clustering.

Sources are compared over their clip pairs rather than as one averaged vector, because a centroid's scale moves
with the clip count, so a threshold calibrated on one run size is wrong on the next. The clip-pair fraction holds
still. Comparing per clip also survives a source that holds more than one voice.
"""

import collections

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

CLIP_PAIR = 0.75                # cosine at which one clip of A and one of B count as agreeing
DUPLICATE = 0.95                # a clip pair this close is the same recording, not a speaker link
BLOCK_BYTES = 1 << 30           # memory one block of the clip-by-clip cosine matrix may take


def compare(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per source pair, in sorted source order: the share of clip pairs that agree, and their closest clip pair.

    The share drives the clustering; the closest pair only tells whether a merge is duplicated audio rather than
    one person recorded twice. All clips are stacked and compared with one matrix product per block of rows,
    then reduced over each source's rows and columns.
    """

    sources = sorted(embeddings)
    counts = np.array([len(embeddings[source]) for source in sources])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    stacked = np.concatenate([embeddings[source] for source in sources])
    agreement = np.empty((len(sources), len(sources)))
    closest = np.empty((len(sources), len(sources)), dtype=np.float32)

    rows = max(int(counts.max()), BLOCK_BYTES // (stacked.shape[0] * 8))
    first = 0
    while first < len(sources):
        last = first + 1
        while last < len(sources) and starts[last] + counts[last] - starts[first] <= rows:
            last += 1
        begin, end = starts[first], starts[last - 1] + counts[last - 1]
        cosines = stacked[begin:end] @ stacked.T
        local = starts[first:last] - begin
        agree = np.add.reduceat(np.add.reduceat((cosines > CLIP_PAIR).astype(np.float32), local, axis=0),
                                starts, axis=1)
        agreement[first:last] = agree / (counts[first:last, None] * counts[None, :])
        closest[first:last] = np.maximum.reduceat(np.maximum.reduceat(cosines, local, axis=0), starts, axis=1)
        first = last
    np.fill_diagonal(agreement, 1.0)
    np.fill_diagonal(closest, 1.0)
    return agreement, closest


def merge_map(sources: list[str], agreement: np.ndarray, threshold: float, method: str = "single") -> dict[str, str]:
    """Source ID -> the lowest ID of its cluster, for the ones that move.

    Single linkage because the errors are not symmetric: merging two similar voices costs a little diversity,
    splitting one person across IDs costs split integrity. Complete linkage splits a person on their weakest
    pair alone.
    """

    if len(sources) < 2:
        return {}
    distance = np.clip(1 - agreement, 0, None)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method=method),
                      1 - threshold, criterion="distance")
    clusters = collections.defaultdict(list)
    for source, label in zip(sources, labels):
        clusters[label].append(source)
    return {source: min(members) for members in clusters.values() if len(members) > 1
            for source in members if source != min(members)}
