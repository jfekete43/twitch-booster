"""Bradley-Terry rating for pairwise map comparisons.

Reference implementation with no third-party dependencies, so it runs
anywhere. At the scale this site operates (a few hundred maps, up to
millions of votes) it is fast enough to use directly; swap in scipy if
full-covariance standard errors are needed for displayed intervals.

Model:  P(i beats j) = sigmoid(theta_i - theta_j)

Fit by maximising the L2-regularised log-likelihood. The penalty is a
zero-mean Gaussian prior: it pins the scale (theta is otherwise identified
only up to an additive constant) and stops maps with very few comparisons
from diverging.
"""

from math import exp, log, sqrt

# Maps below this many comparisons are left unranked on the leaderboard
# rather than shown at a misleadingly precise position.
MIN_COMPARISONS = 30

DEFAULT_LAMBDA = 1.0
ELO_SCALE = 400.0 / log(10.0)


def sigmoid(x):
    """Logistic function, written to avoid overflow at large |x|."""
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


def fit(comparisons, map_ids=None, lam=DEFAULT_LAMBDA, tol=1e-9, max_iter=500):
    """Estimate latent strengths from pairwise outcomes.

    comparisons: iterable of (winner_id, loser_id). Skips must already be
                 filtered out -- a skip carries no information about which
                 map is better.
    map_ids:     optional full universe of ids. Pass it so that maps with
                 zero comparisons still appear in the output (at theta 0,
                 with infinite uncertainty) rather than silently vanishing.

    Returns {map_id: {theta, stderr, n_comparisons, n_wins}}.
    """
    comparisons = list(comparisons)

    ids = set(map_ids or ())
    for w, l in comparisons:
        ids.add(w)
        ids.add(l)
    ids = sorted(ids)
    idx = {m: i for i, m in enumerate(ids)}
    n = len(ids)

    pairs = [(idx[w], idx[l]) for w, l in comparisons]

    n_comp = [0] * n
    n_wins = [0] * n
    for w, l in pairs:
        n_comp[w] += 1
        n_comp[l] += 1
        n_wins[w] += 1

    theta = [0.0] * n

    for _ in range(max_iter):
        grad = [0.0] * n
        fisher = [0.0] * n

        for w, l in pairs:
            p = sigmoid(theta[w] - theta[l])
            resid = 1.0 - p          # observed (1) minus expected
            grad[w] += resid
            grad[l] -= resid
            info = p * (1.0 - p)
            fisher[w] += info
            fisher[l] += info

        # Gaussian prior contributes -lam*theta to the gradient and +lam to
        # the information. It also guarantees fisher > 0, so the Newton step
        # below is always well defined -- including for maps with no votes.
        delta_max = 0.0
        for i in range(n):
            grad[i] -= lam * theta[i]
            fisher[i] += lam
            step = grad[i] / fisher[i]
            theta[i] += step
            if abs(step) > delta_max:
                delta_max = abs(step)

        # Re-centre: only differences in theta are identified.
        mean = sum(theta) / n
        for i in range(n):
            theta[i] -= mean

        if delta_max < tol:
            break

    # Final information pass for standard errors. This is the independent
    # approximation (diagonal only) -- it understates the true standard error
    # slightly because it ignores off-diagonal covariance. Adequate for
    # ordering maps by uncertainty, which is what the sampler needs. For
    # displayed confidence intervals, invert the full Fisher matrix instead.
    fisher = [lam] * n
    for w, l in pairs:
        p = sigmoid(theta[w] - theta[l])
        info = p * (1.0 - p)
        fisher[w] += info
        fisher[l] += info

    return {
        m: {
            "theta": theta[i],
            "stderr": 1.0 / sqrt(fisher[i]),
            "n_comparisons": n_comp[i],
            "n_wins": n_wins[i],
        }
        for m, i in idx.items()
    }


def display_rating(theta):
    """Elo-shaped rating for humans. 1847 reads as a rating; 0.84 does not."""
    return round(1500.0 + theta * ELO_SCALE)


def rank(ratings, min_comparisons=MIN_COMPARISONS):
    """Order maps by strength, leaving under-measured ones unranked.

    Returns a list of dicts sorted strongest first. Maps below the comparison
    threshold sort last and carry rank None.
    """
    rows = [
        {
            "map_id": m,
            "theta": r["theta"],
            "display": display_rating(r["theta"]),
            "stderr": r["stderr"],
            "n_comparisons": r["n_comparisons"],
            "n_wins": r["n_wins"],
            "ranked": r["n_comparisons"] >= min_comparisons,
        }
        for m, r in ratings.items()
    ]
    rows.sort(key=lambda r: (not r["ranked"], -r["theta"]))

    position = 0
    for row in rows:
        if row["ranked"]:
            position += 1
            row["rank"] = position
        else:
            row["rank"] = None
    return rows


def pair_information(theta_i, theta_j, var_i, var_j):
    """Expected information from comparing two maps. Higher is better.

    Fisher information from one Bradley-Terry comparison is p*(1-p), maximised
    at p = 0.5: you learn most from matchups you cannot predict. Scaling by
    current variance prioritises maps that are still poorly measured, which is
    what makes newly added maps converge without any special-casing.
    """
    p = sigmoid(theta_i - theta_j)
    return p * (1.0 - p) * (var_i + var_j)
