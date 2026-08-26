import collections
import math
import random

import numpy as np

from tools.prosody.constants import (
    DEFAULT_CEILING,
    DEFAULT_FLOOR,
    DEFAULT_SPEAKER_CAP,
    EXPR_EDGES,
    PITCH_EDGE,
    RATE_BINS,
    SLOPE_LEVEL,
)


def assign_cells(rows):
    """
    Stratify on three named axes: expressiveness x speaking rate x phrase-final direction.

    Expressiveness is cut at percentiles [10, 35, 65, 90] rather than quartiles so the extreme
    deciles become cells of their own. Quantile bins cannot flatten the feature they are cut
    from — a pool is uniform over its own quantiles by construction.

    Edges are percentiles of the rows passed in, so call this on the whole pool: re-running it
    on a selected subset re-cuts the edges and reports different cells for the same clips.
    """

    expressiveness = np.array([row["st_range"] for row in rows])
    rate = np.array([row["voiced_onsets_per_s"] for row in rows])
    final = np.array([row["st_final_slope"] for row in rows])

    expr_bin = np.digitize(expressiveness, np.percentile(expressiveness, EXPR_EDGES))
    rate_bin = np.digitize(
        rate, np.percentile(rate, np.linspace(0, 100, RATE_BINS + 1)[1:-1])
    )
    direction = np.where(final < -SLOPE_LEVEL, 0, np.where(final > SLOPE_LEVEL, 2, 1))
    return [(int(e), int(r), int(d)) for e, r, d in zip(expr_bin, rate_bin, direction)]


def water_fill(capacity, budget, rng):
    """
    Maximum-entropy allocation: an equal quota per cell, with the surplus of cells that cannot
    fill it redistributed over those that can.

    Ties are broken randomly. With every cell wanting the same amount, an index-ordered
    tie-break would spend the whole budget on one contiguous corner of the grid.
    """

    take = np.zeros(len(capacity), dtype=int)
    remaining, active = budget, capacity > 0
    while remaining > 0 and active.any():
        quota = max(1, remaining // active.sum())
        give = np.minimum(quota, capacity - take) * active
        if not give.sum():
            break
        for index in np.argsort(-give + rng.random(len(give)) * 1e-6):
            if remaining <= 0:
                break
            granted = min(give[index], remaining)
            take[index] += granted
            remaining -= granted
        active &= take < capacity
    return take


def balanced_cells(cells, capacity, budget):
    """
    Choose which cells get a clip when the budget is smaller than the number of cells.

    Water-fill can only say "one clip each" there and leaves the rest to a tie-break; taking
    the cell whose bin values are so far least used keeps every marginal balanced instead.
    """

    used = [collections.Counter() for _ in range(len(cells[0]))]
    chosen, available = [], [i for i, c in enumerate(capacity) if c > 0]
    while len(chosen) < budget and available:
        index = min(
            available,
            key=lambda i: (
                sum(used[axis][cells[i][axis]] for axis in range(len(used))),
                -capacity[i],
            ),
        )
        chosen.append(index)
        for axis in range(len(used)):
            used[axis][cells[index][axis]] += 1
        available.remove(index)

    quota = np.zeros(len(capacity), dtype=int)
    quota[chosen] = 1
    return quota


def select_diverse(rows, budget, speaker_cap=DEFAULT_SPEAKER_CAP, seed=0):
    """
    Select `budget` rows whose prosody-cell histogram is as flat as the data allows.

    Rows are the output of prosody_features plus a "speaker" key. Cells are filled by
    water-filling an equal quota, then drawn uniformly inside each cell under a per-speaker
    cap. Uniform is deliberate: greedy facility-location picks each cell's centre, which
    measured *below* random on tail coverage and biased selection toward long clips.

    Keep speaker_cap * n_speakers comfortably above budget. Once the cap cannot supply the
    budget it, rather than the stratification, decides the subset. The result can fall a few
    short of budget when the cap empties a cell that still holds quota.
    """

    rng = np.random.default_rng(seed)
    members = collections.defaultdict(list)
    for row, cell in zip(rows, assign_cells(rows)):
        members[cell].append(row)

    cells = sorted(members)
    capacity = np.array(
        [
            sum(
                min(speaker_cap, n)
                for n in collections.Counter(r["speaker"] for r in members[cell]).values()
            )
            for cell in cells
        ]
    )
    quota = (
        balanced_cells(cells, capacity, budget)
        if budget <= len(cells)
        else water_fill(capacity, budget, rng)
    )

    taken = collections.Counter()
    selected = []
    for cell, k in sorted(zip(cells, quota), key=lambda item: -item[1]):
        picked = 0
        for index in rng.permutation(len(members[cell])):
            if picked == k:
                break
            row = members[cell][index]
            if taken[row["speaker"]] < speaker_cap:
                selected.append(row)
                taken[row["speaker"]] += 1
                picked += 1
    return selected


def _entropy(counts, total):
    return math.log2(total) - sum(n * math.log2(n) for n in counts.values()) / total


def select_bounded(rows, floor=DEFAULT_FLOOR, ceiling=DEFAULT_CEILING, seed=0, high_bounds=None):
    """
    Select rows under a per-speaker floor and ceiling, taking as much data as stays flat.

    Every speaker holding at least `floor` rows contributes exactly that many, spread over
    their rarest cells; speakers below the floor are dropped. Beyond the floor, rows are drawn
    from the thinnest cell for as long as each one strictly raises the cell histogram's
    entropy, so a speaker approaches `ceiling` only while their rows still land where the
    histogram is short.

    Unlike select_diverse there is no budget: the draw stops when nothing flattens further.

    `high_bounds` is an optional (floor, ceiling) applied instead to speakers whose reference
    pitch is above PITCH_EDGE, which oversamples high voices. Only its floor is a guarantee:
    reference pitch is not one of the cell axes, so a raised ceiling merely permits growth
    the entropy gate may still refuse.
    """

    rng = random.Random(seed)
    for row, cell in zip(rows, assign_cells(rows)):
        row["cell"] = cell

    by_speaker = collections.defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    bounds = {
        speaker: high_bounds if high_bounds and clips[0]["ref_hz"] > PITCH_EDGE else (floor, ceiling)
        for speaker, clips in by_speaker.items()
    }
    rarity = collections.Counter(row["cell"] for row in rows)

    taken, counts, selected = collections.Counter(), collections.Counter(), []
    for speaker, clips in sorted(by_speaker.items()):
        speaker_floor = bounds[speaker][0]
        if len(clips) < speaker_floor:
            continue
        buckets = collections.defaultdict(list)
        for row in clips:
            buckets[row["cell"]].append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        order = sorted(buckets, key=lambda cell: rarity[cell])
        while taken[speaker] < speaker_floor:
            for cell in order:
                if buckets[cell] and taken[speaker] < speaker_floor:
                    row = buckets[cell].pop()
                    selected.append(row)
                    counts[row["cell"]] += 1
                    taken[speaker] += 1

    chosen = {id(row) for row in selected}
    available = collections.defaultdict(list)
    for row in rows:
        if id(row) not in chosen and taken[row["speaker"]]:
            available[row["cell"]].append(row)
    for bucket in available.values():
        rng.shuffle(bucket)

    total = len(selected)
    weight = sum(n * math.log2(n) for n in counts.values())
    while True:
        live = [cell for cell in available if available[cell]]
        if not live:
            break
        cell = min(live, key=lambda cell: counts[cell])
        n = counts[cell]
        gain = (n + 1) * math.log2(n + 1) - (n * math.log2(n) if n else 0)
        if math.log2(total + 1) - (weight + gain) / (total + 1) <= _entropy(counts, total):
            break
        row = None
        while available[cell] and row is None:
            candidate = available[cell].pop()
            if taken[candidate["speaker"]] < bounds[candidate["speaker"]][1]:
                row = candidate
        if row is None:
            continue
        selected.append(row)
        counts[cell] += 1
        taken[row["speaker"]] += 1
        total += 1
        weight += gain
    return selected
