"""Exercise the points pipeline against the real map database.

Run:  python3 rating/validate_scoring.py

Simulates submissions to check the scoring, filtering and Bradley-Terry
aggregation all work end to end on the actual 262-map dataset.

The simulation gives every map a hidden "quality" that is independent of
its game's reach, and gives every voter a set of games they have played,
weighted by how widely played that game actually was. That separation is
deliberate: it lets the run report how much of the resulting leaderboard
is driven by audience size rather than by the maps themselves.

The numbers below are synthetic. The mechanism they illustrate is not.
"""

import json
import os
import random
from collections import Counter

from scoring import score_lists, leaderboard, aggregate_comparisons, POINTS
from bradley_terry import fit, rank

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data", "maps.json")

# Rough relative reach of each game's player base. Not precise figures --
# just enough spread to show how reach propagates into the ranking.
REACH = {
    "cod4": 1.00, "blackops": 1.00, "halo3": 0.95, "cs": 0.95,
    "tf2": 0.70, "halo2": 0.65, "gears1": 0.55, "r6siege": 0.75,
    "destiny": 0.45, "r6vegas": 0.30, "socom2": 0.18,
    "killzone2": 0.15, "rfom": 0.15,
}

LIST_SIZE = 9          # the 3x3 grid; set to 10 for the list format
N_SUBMISSIONS = 5000


def main():
    rng = random.Random(7)
    db = json.load(open(DATA))

    games = {g["id"]: g for g in db["games"]}
    maps = {
        m["id"]: {
            "name": m["name"],
            "game_id": m["game_id"],
            "game": games[m["game_id"]]["short"],
            "franchise": games[m["game_id"]]["franchise"],
        }
        for m in db["maps"]
    }
    all_ids = list(maps)
    by_game = {}
    for mid, meta in maps.items():
        by_game.setdefault(meta["game_id"], []).append(mid)

    # Hidden quality, independent of reach.
    quality = {mid: rng.gauss(0, 1) for mid in all_ids}

    # Each voter knows a subset of games, sampled by reach; they pick their
    # nine from what they know, favouring higher-quality maps.
    lists = []
    for _ in range(N_SUBMISSIONS):
        known = [g for g in by_game if rng.random() < REACH.get(g, 0.3)]
        if not known:
            continue
        pool = [m for g in known for m in by_game[g]]
        if len(pool) < LIST_SIZE:
            continue
        scored = sorted(pool, key=lambda m: quality[m] + rng.gauss(0, 1.2), reverse=True)
        lists.append(scored[:LIST_SIZE])

    scores = score_lists(lists, map_ids=all_ids)
    board = leaderboard(scores, maps)

    print(f"{len(lists):,} simulated top-{LIST_SIZE} submissions over "
          f"{len(all_ids)} maps")
    print(f"points table: {POINTS[:LIST_SIZE]}\n")

    print("TOP 15 BY POINTS")
    print(f"  {'#':>3}  {'map':<22} {'game':<12} {'points':>8} {'lists':>7} {'#1s':>6}")
    for r in board[:15]:
        print(f"  {r['rank']:>3}  {r['name']:<22} {r['game']:<12} "
              f"{r['points']:>8,} {r['appearances']:>7,} {r['n_first']:>6,}")

    # --- filtering -------------------------------------------------------
    halo = leaderboard(scores, maps, franchise="Halo")
    socom = leaderboard(scores, maps, game="socom2")
    print(f"\nFILTERS")
    print(f"  Halo franchise: {len(halo)} maps; best is {halo[0]['name']} "
          f"(overall #{halo[0]['global_rank']})")
    print(f"  SOCOM 2 only:   {len(socom)} maps; best is {socom[0]['name']} "
          f"(overall #{socom[0]['global_rank']})")

    # --- what drives the ordering ---------------------------------------
    top50 = Counter(r["franchise"] for r in board[:50])
    print("\nFRANCHISE SHARE OF TOP 50")
    for f, n in top50.most_common():
        print(f"  {f:<16} {n:>2}")

    unranked = [r for r in board if r["appearances"] == 0]
    print(f"\n  maps appearing in zero lists: {len(unranked)}")

    # How much of the outcome is reach rather than quality?
    ranked = [r for r in board if r["appearances"] > 0]
    by_quality = sorted(ranked, key=lambda r: -quality[r["map_id"]])
    true_pos = {r["map_id"]: i for i, r in enumerate(by_quality)}
    drift = sum(abs(true_pos[r["map_id"]] - i) for i, r in enumerate(ranked))
    drift /= max(len(ranked), 1)
    print(f"  mean rank drift from hidden quality order: {drift:.1f} places")

    low_reach = [g for g, v in REACH.items() if v <= 0.2]
    best_low = min((r["rank"] for r in board
                    if r["game_id"] in low_reach and r["appearances"] > 0),
                   default=None)
    print(f"  best-placed map from a low-reach game "
          f"({', '.join(games[g]['short'] for g in low_reach)}): #{best_low}")

    # --- Bradley-Terry over the same submissions -------------------------
    triples = aggregate_comparisons(lists, all_ids, flat_losses=True)
    implied = sum(k for _, _, k in triples)
    print(f"\nBRADLEY-TERRY VIEW (flat losses)")
    print(f"  {implied:,} implied comparisons collapsed into "
          f"{len(triples):,} distinct pairs")

    bt = rank(fit(triples, map_ids=all_ids), min_comparisons=1)
    print("  top 5 by rating:")
    for r in bt[:5]:
        print(f"    {r['rank']:>2}. {maps[r['map_id']]['name']:<22} "
              f"{r['display']:>5}")

    print("\nPIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
