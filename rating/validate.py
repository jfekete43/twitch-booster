"""Validation for the Bradley-Terry fit, on synthetic data with known answers.

Run:  python3 rating/validate.py

Checks three things, the last two being properties the design argument
actually rests on:

  1. The fit recovers a known ranking from noisy pairwise votes.
  2. The fit is order-independent -- shuffling the votes changes nothing.
  3. A map added late is not penalised for arriving late.

(2) and (3) are exactly where Elo fails and are the reason this site uses a
batch Bradley-Terry fit instead.
"""

import random
from bradley_terry import fit, sigmoid, display_rating, rank


def spearman(a, b):
    """Rank correlation between two equal-length sequences."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db)


def simulate(true_theta, n_votes, rng, pool=None):
    """Draw random matchups and resolve them by the Bradley-Terry model."""
    ids = pool if pool is not None else list(true_theta)
    out = []
    for _ in range(n_votes):
        i, j = rng.sample(ids, 2)
        p = sigmoid(true_theta[i] - true_theta[j])
        out.append((i, j) if rng.random() < p else (j, i))
    return out


def main():
    rng = random.Random(42)
    ok = True

    # ---- 1. recovery -------------------------------------------------
    n_maps = 262                      # matches the real dataset size
    true_theta = {f"m{i}": rng.gauss(0, 1) for i in range(n_maps)}
    votes = simulate(true_theta, 60_000, rng)

    ratings = fit(votes, map_ids=list(true_theta))
    ids = sorted(true_theta)
    rho = spearman([true_theta[m] for m in ids], [ratings[m]["theta"] for m in ids])

    print(f"1. recovery over {n_maps} maps, {len(votes):,} votes")
    print(f"   spearman vs. ground truth: {rho:.4f}")
    if rho < 0.95:
        print("   FAIL: expected > 0.95")
        ok = False
    else:
        print("   pass")

    # ---- 2. order independence --------------------------------------
    shuffled = votes[:]
    rng.shuffle(shuffled)
    reordered = fit(shuffled, map_ids=list(true_theta))
    drift = max(abs(ratings[m]["theta"] - reordered[m]["theta"]) for m in ids)

    print("\n2. order independence (same votes, shuffled)")
    print(f"   max theta difference: {drift:.2e}")
    if drift > 1e-6:
        print("   FAIL: fit depends on vote order")
        ok = False
    else:
        print("   pass -- identical ratings regardless of vote order")

    # ---- 3. no penalty for arriving late -----------------------------
    # A genuinely strong map added after the leaderboard is established.
    # Under Elo it would have to climb from a default rating; under batch BT
    # it should land near its true strength immediately.
    incumbents = [f"m{i}" for i in range(50)]
    base_theta = {m: true_theta[m] for m in incumbents}
    history = simulate(base_theta, 20_000, rng, pool=incumbents)

    late_id, late_theta = "late_arrival", 2.0     # stronger than almost everything
    true_position = 1 + sum(1 for t in base_theta.values() if t > late_theta)

    def late_matchups(count):
        out = []
        for _ in range(count):
            opp = rng.choice(incumbents)
            p = sigmoid(late_theta - base_theta[opp])
            out.append((late_id, opp) if rng.random() < p else (opp, late_id))
        return out

    def place(count):
        combined = fit(history + late_matchups(count),
                       map_ids=incumbents + [late_id])
        rows = rank(combined, min_comparisons=30)
        return next(r for r in rows if r["map_id"] == late_id)

    # 3a. Too few comparisons: the prior should shrink it toward the mean and
    # the leaderboard should withhold a rank. Both are intended behaviour.
    sparse = place(15)
    print(f"\n3a. late arrival, only 15 comparisons "
          f"(vs. {len(history):,} of prior history)")
    print(f"    theta {sparse['theta']:.3f} (true {late_theta:.3f}), "
          f"rank {sparse['rank']}")
    if sparse["rank"] is None and sparse["theta"] < late_theta:
        print("    pass -- shrunk toward the mean and left unranked, by design")
    else:
        print("    FAIL: expected shrinkage and no rank below MIN_COMPARISONS")
        ok = False

    # 3b. Enough comparisons: it should land on merit, with no climb.
    placed = place(120)
    print("\n3b. late arrival, 120 comparisons")
    print(f"    estimated theta: {placed['theta']:.3f}  (true {late_theta:.3f})")
    print(f"    display rating:  {display_rating(placed['theta'])}")
    print(f"    placed at rank:  {placed['rank']}  (true rank {true_position})")
    if abs(placed["theta"] - late_theta) > 0.5:
        print("    FAIL: late arrival not recovered near its true strength")
        ok = False
    else:
        print("    pass -- no climb required, placed on merit immediately")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
