import itertools

import numpy as np

from resonance.speakers import clustering


def reference(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The pair-by-pair loop the vectorised compare replaced."""

    speakers = sorted(embeddings)
    agreement, closest = np.eye(len(speakers)), np.ones((len(speakers),) * 2)
    for i, j in itertools.combinations(range(len(speakers)), 2):
        cosines = embeddings[speakers[i]] @ embeddings[speakers[j]].T
        agreement[i, j] = agreement[j, i] = float((cosines > clustering.CLIP_PAIR).mean())
        closest[i, j] = closest[j, i] = float(cosines.max())
    return agreement, closest


def population(seed: int = 0) -> dict[str, np.ndarray]:
    """40 sources of 1-4 clips over 12 people, so several sources share a voice."""

    rng = np.random.default_rng(seed)
    people = rng.normal(size=(12, 64))
    embeddings = {}
    for index in range(40):
        vectors = people[index % 12] + rng.normal(scale=0.35, size=(rng.integers(1, 5), 64))
        embeddings[f"video{index:02d}"] = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    return embeddings


def test_compare_matches_the_pairwise_loop(monkeypatch):
    embeddings = population()
    expected = reference(embeddings)
    for block in (1 << 30, 4096):       # one block, then many small ones
        monkeypatch.setattr(clustering, "BLOCK_BYTES", block)
        agreement, closest = clustering.compare(embeddings)
        assert np.allclose(agreement, expected[0]) and np.allclose(closest, expected[1], atol=1e-6)


def test_merge_map_groups_the_same_person():
    embeddings = population()
    agreement, _ = clustering.compare(embeddings)
    merges = clustering.merge_map(sorted(embeddings), agreement, threshold=0.5)
    assert merges and all(int(source[5:]) % 12 == int(target[5:]) % 12 for source, target in merges.items())
    assert all(target < source for source, target in merges.items())
    assert clustering.merge_map(["only"], np.eye(1), 0.5) == {}
