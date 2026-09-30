"""Points scoring for submitted top-N map lists.

The visible ranking is a straight points total: every map in a submitted
list earns points by its position, and the leaderboard is the sum across
all submissions. Simple, instantly computed, and verifiable by hand --
which is most of why people trust a leaderboard.

Not being picked scores nothing, which is the same thing as losing to
everything that was picked. The flat-loss model is therefore already
baked into the points; it only needs explicit handling for the
Bradley-Terry view (see aggregate_comparisons).

List length is not fixed. POINTS carries ten values so the same table
serves a top-9 grid or a top-10 list -- a top-9 uses the first nine.
"""

from collections import defaultdict

# F1-shaped rather than linear. Being someone's #1 is a far stronger signal
# than being their #9, and linear weighting (9,8,7...) understates that.
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def points_for(position, table=POINTS):
    """Points for a 1-indexed position. Positions past the table score 0."""
    if position < 1 or position > len(table):
        return 0
    return table[position - 1]


def score_lists(lists, map_ids=None, table=POINTS):
    """Aggregate submitted lists into per-map totals.

    lists:    iterable of ordered map-id sequences, best first.
    map_ids:  optional full universe, so maps nobody picked still appear
              (at zero points) rather than vanishing from the leaderboard.

    Returns {map_id: {points, appearances, n_first, avg_position}}.
    """
    points = defaultdict(int)
    appearances = defaultdict(int)
    n_first = defaultdict(int)
    position_sum = defaultdict(int)

    for entries in lists:
        for i, map_id in enumerate(entries):
            position = i + 1
            points[map_id] += points_for(position, table)
            appearances[map_id] += 1
            position_sum[map_id] += position
            if position == 1:
                n_first[map_id] += 1

    universe = set(map_ids or ()) | set(appearances)
    return {
        m: {
            "points": points[m],
            "appearances": appearances[m],
            "n_first": n_first[m],
            "avg_position": (position_sum[m] / appearances[m]) if appearances[m] else None,
        }
        for m in universe
    }


def leaderboard(scores, maps, franchise=None, game=None):
    """Rank maps by points, optionally filtered to a franchise or a game.

    maps: {map_id: {"name","game_id","game","franchise"}} for display.

    Global rank is computed before filtering and carried through, so a
    filtered view can show both "3rd best Halo map" and "14th overall".
    """
    rows = []
    for map_id, s in scores.items():
        meta = maps.get(map_id, {})
        rows.append({
            "map_id": map_id,
            "name": meta.get("name", map_id),
            "game_id": meta.get("game_id"),
            "game": meta.get("game"),
            "franchise": meta.get("franchise"),
            **s,
        })

    # Ties break on appearances, then on number of first-place picks: a map
    # reaching the same total from more lists is the better-supported result.
    rows.sort(key=lambda r: (-r["points"], -r["appearances"], -r["n_first"], r["name"]))
    for i, row in enumerate(rows):
        row["global_rank"] = i + 1

    if franchise:
        rows = [r for r in rows if r["franchise"] == franchise]
    if game:
        rows = [r for r in rows if r["game_id"] == game]

    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def within_list_pairs(entries):
    """Every ordered pair implied by one list: earlier beats later.

    A ranked list of 9 yields 36 comparisons, a list of 10 yields 45. This
    is the signal an unordered grid throws away, and the reason the tiles
    are numbered.
    """
    return [
        (entries[i], entries[j])
        for i in range(len(entries))
        for j in range(i + 1, len(entries))
    ]


def aggregate_comparisons(lists, all_map_ids, flat_losses=True):
    """Collapse submissions into weighted pairwise win counts for the BT fit.

    Returns [(winner, loser, count), ...].

    Materialising flat losses one row per comparison does not scale: with
    262 maps, each top-9 submission implies 9 * 253 = 2,277 of them. But the
    number of *distinct* pairs is bounded by the map count regardless of how
    many lists are submitted, so aggregating into counts keeps the fit the
    same size whether there are a thousand submissions or ten million.
    """
    universe = set(all_map_ids)
    counts = defaultdict(int)

    for entries in lists:
        picked = [m for m in entries if m in universe]
        for w, l in within_list_pairs(picked):
            counts[(w, l)] += 1

        if flat_losses:
            unpicked = universe - set(picked)
            for w in picked:
                for l in unpicked:
                    counts[(w, l)] += 1

    return [(w, l, c) for (w, l), c in counts.items()]
