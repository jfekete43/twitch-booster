# Map ranking site — technical design

A pairwise voting site. Visitors are shown two multiplayer maps, pick the better
one, and those votes feed a Bradley-Terry model that produces an all-time
ranking across every game in the database.

Current dataset: 262 maps across 13 games (`data/maps.json`).

---

## 1. Core principles

Three decisions drive most of the design. They're recorded here because the
reasons are non-obvious and easy to undo by accident later.

**Votes are append-only; ratings are derived.** The `votes` table is the source
of truth and is never mutated. `ratings` is a materialized view rebuilt from
scratch on every fit. This is what makes it possible to discover a brigading
session next month and refit the entire history with its votes excluded. If
ratings were updated incrementally in place, that correction would be
impossible without losing everything.

**Familiarity is a property of maps, not games.** There is no up-front game
selection screen. Voters skip pairs they don't know, and the sampler infers
what they're familiar with from their behaviour. A checkbox gate walls off
smaller games and lets a narrow enthusiast pool inflate them; it also throws
away the skip signal, which is useful data in its own right.

**Cross-game pairs are structural, not decorative.** Votes form a graph where
maps are nodes and comparisons are edges. If maps from different games are
never compared, the graph has disconnected components and ratings are only
meaningful *within* a component — you'd have 13 separate rankings, not one
list. Cross-game pairs are the bridges that put everything on a single scale.

---

## 2. Data model

```
games
  id            text primary key      -- 'halo3'
  name          text                  -- 'Halo 3'
  short         text                  -- display label
  year          int
  developer     text
  platform      text

maps
  id            text primary key      -- 'halo3-narrows'
  game_id       text references games
  name          text
  release       text                  -- base | dlc | canonical | lifetime
  dlc_pack      text null
  remake_of     text null references maps   -- halo3-blackout -> halo2-lockout
  image_url     text null
  thumb_url     text null
  active        bool default true     -- false hides without deleting

sessions
  id            uuid primary key
  created_at    timestamptz
  last_seen     timestamptz
  vote_count    int default 0
  ip_hash       text                  -- salted hash, never the raw address
  ua_hash       text
  excluded      bool default false    -- set on detected brigading; refit drops these

votes                                 -- APPEND ONLY
  id            bigserial primary key
  session_id    uuid references sessions
  map_a         text references maps
  map_b         text references maps
  winner        text null             -- map id, or NULL for a skip
  skip_reason   text null             -- 'unfamiliar' (later: 'tie')
  pair_kind     text                  -- 'within_game' | 'cross_game'
  ms_to_decide  int
  created_at    timestamptz

ratings                               -- DERIVED, rebuilt by the fit job
  map_id        text primary key references maps
  theta         double precision      -- latent strength, centred at 0
  stderr        double precision
  display       int                   -- Elo-scaled, for humans
  n_comparisons int
  n_wins        int
  rank          int null              -- null until n_comparisons >= MIN_COMPARISONS
  prev_rank     int null
  updated_at    timestamptz
```

Indexes that matter: `votes(map_a)`, `votes(map_b)`, `votes(session_id)`,
`votes(created_at)`, and `maps(game_id)`.

A skip is stored as a vote row with `winner IS NULL`. It is excluded from the
fit but retained — the skip rate per map is a free obscurity metric and drives
both the sampler and the "most underrated map" content later.

---

## 3. Rating system

### Model

Bradley-Terry. Each map has a latent strength `θ`, and

```
P(i beats j) = σ(θᵢ - θⱼ)        where σ(x) = 1 / (1 + e^(-x))
```

Bradley-Terry rather than Elo, because maps are *static*. Elo is a streaming
approximation designed for players whose skill drifts: it's order-dependent,
has an arbitrary K-factor, and makes a late-added map climb from a default
rating. Batch BT re-estimates every map jointly from the complete vote history,
so arrival order is irrelevant and a map added next year is measured on the
same footing as one present at launch.

### Objective

Maximise the regularised log-likelihood:

```
L(θ) = Σ log σ(θ_w - θ_l)  -  (λ/2) Σ θᵢ²
```

The L2 term is a zero-mean Gaussian prior (MAP estimation). It does two jobs:
it pins the scale (θ is otherwise only identified up to an additive constant),
and it stops maps with very few comparisons from diverging — without it, a map
that won its only matchup has infinite strength.

`λ = 1.0` is a reasonable default. Higher shrinks under-measured maps harder
toward the mean.

### Fitting

Diagonal Newton, iterated to convergence:

```
gradᵢ   = Σ (y - p)  over comparisons involving i,  minus  λθᵢ
fisherᵢ = Σ p(1-p)   over comparisons involving i,  plus   λ
θᵢ     += gradᵢ / fisherᵢ
```

Then re-centre so `Σθ = 0`.

At 262 maps this converges in well under a second even in pure Python, so the
fit is not a performance concern at any realistic vote volume.

### Uncertainty

`stderr ≈ 1/√fisherᵢ`. This is the independent approximation — it ignores
off-diagonal covariance and so slightly understates the true standard error. It
is fine for ranking maps by uncertainty (which is all the sampler needs) but if
you display confidence intervals, invert the full Fisher matrix instead
(`scipy.linalg.inv`).

### Display scale

`display = 1500 + θ · (400 / ln 10)`

Elo-shaped on purpose. The audience is gamers; 1847 reads as a rating, 0.84
reads as nothing.

### Refit cadence

Batch job on a cron, every 5–15 minutes. Not per-vote. Votes are cheap appends;
the fit is the expensive part and does not need to be live.

If a number that moves instantly is wanted for feedback, run Elo as a display
layer on the client and treat the batch BT fit as canonical. Do not let the two
diverge in the database.

---

## 4. Pair selection

The single highest-leverage component. Uniform random pairing wastes votes and
converges slowly.

### Information gain

For Bradley-Terry, the Fisher information contributed by one comparison of i
and j is `p(1-p)` where `p = σ(θᵢ-θⱼ)`. This is maximised at `p = 0.5` — you
learn most from pairs you can't predict. Weighting by current uncertainty
prioritises maps that are still poorly measured:

```
score(i,j) = p(1-p) · (varᵢ + varⱼ)
```

This solves the exposure problem structurally. New and under-sampled maps have
the widest intervals, so they get oversampled automatically until they reach
parity — no special-casing needed when a game is added.

### Composition

```
65%  within-game pairs
35%  cross-game pairs        -- the bridges; non-negotiable
```

Within the chosen type, sample from the top-scoring candidates with some
randomness rather than taking the argmax — pure information-maximising pairing
is repetitive and no fun.

When a new game is added, deliberately weight its maps toward *high-certainty*
incumbents for the first few hundred comparisons. This is both the fairest and
the most efficient choice: a comparison against a precisely-measured opponent
pins down the newcomer fastest.

### Familiarity inference

Replaces the checkbox gate. Per session, per game, track votes and skips as a
Beta posterior:

```
familiarity(game) = (α + votes) / (α + votes + β + skips)
```

with a weak prior (`α = β = 1`) so it adapts within roughly ten votes. Multiply
candidate scores by the familiarity estimate for each map's game.

**Keep 20% pure exploration** — pairs sampled ignoring familiarity entirely.
Without it the sampler converges on a voter's two favourite games and rebuilds
the walled gardens the checkbox system would have created.

---

## 5. Vote integrity

This is a contentious-topic voting site on the internet. Fanbases will organise.

- Rate limit per session and per `ip_hash`.
- Daily vote cap per session.
- Reject votes with implausible `ms_to_decide` (< ~250ms is not a human
  judgement about two images).
- Flag sessions whose votes are overwhelmingly concentrated in one game, or
  whose timing distribution is machine-like.
- Set `sessions.excluded = true` and refit. Because votes are append-only and
  ratings are derived, this is a clean operation with no data loss — and it is
  reversible if the flag turns out to be wrong.

Store hashes, never raw IPs or user agents.

---

## 6. Routes

```
GET  /                     vote page
GET  /api/pair             fetch next pair (used for preloading)
POST /api/vote             record vote, return the following pair
GET  /rankings             global leaderboard
GET  /rankings/:game       per-game leaderboard
GET  /map/:id              map detail: rating, record, head-to-head
GET  /my-rankings          this session's picks vs. consensus
```

### Preloading is the whole experience

Serve pair N+1 in the response to vote N, and keep a client-side queue of two
to three pairs with their images already loaded. A 400ms image load between
votes makes a rapid-fire voting app feel broken, and this is the single
largest factor in how the site feels. Design the API around it from the start
rather than retrofitting.

### Leaderboard display

Maps with `n_comparisons < MIN_COMPARISONS` (default 30) get `rank = NULL` and
render as "still measuring" rather than at a misleadingly precise position.
Early on the shrinkage from the prior will pull under-measured maps toward the
middle; showing them as ranked would read as a real result rather than an
absence of data.

---

## 7. Scope

**v1**
- Vote loop with preloading
- "Haven't played one/both" skip
- Bradley-Terry fit on a cron
- Global and per-game leaderboards
- No accounts — session cookie only

**Later**
- Personal rankings vs. consensus (strong share unit, free from data already
  collected)
- Most divisive map — closest to 50/50 across many comparisons
- Daily set of ten matchups, Wordle-shaped, to create a return loop and cap
  ballot stuffing
- Rank movement over time; annual versioned lists

### Versioning the list

Publish as "the 2026 list" rather than implying objective truth. The ranking
reflects who showed up to vote, and saying so is both more honest and better
content. It also creates a reason to publish annually, and year-over-year
movement is its own story.

---

## 8. Known biases

**Recognition bias** is the largest validity threat. People vote for the map
they recognise, not the one that is better. Mitigated by always showing
screenshots rather than names alone, and by the skip button removing forced
guesses — but not eliminated.

**Visual inconsistency is a confound, not a cosmetic problem.** If one map has
a 4K remaster render and its opponent has a muddy 2004 JPEG, the vote measures
image quality rather than map quality. Source images at consistent resolution,
aspect ratio and HUD state. Consider a layout/overhead view as a secondary
image for exactly this reason — you're rating design, not graphics.

**Audience composition** cannot be fixed by the estimator. Early voters
self-select on the games they already love, and cross-game bridges are judged
only by people who know both. This is measurable, though: compare how a game's
maps perform against incumbents versus what BT predicts from their within-game
ratings. A systematic gap is the size of the nostalgia effect.

**Remakes.** Halo 3's Blackout is Halo 2's Lockout; Heretic is Midship. They
are kept as separate entries with a `remake_of` link rather than merged,
because they play differently in different sandboxes and the comparison is a
real argument. Expect them to land at different ranks, and consider surfacing
the relationship in the UI when such a pair comes up.

---

## 9. Open decisions

| Decision | Status |
|---|---|
| Stack and hosting | **needed before implementation** |
| Image sourcing and licensing | blocked on network access; critical path to launch |
| Tie / "too close to call" as a third button | deferred; modellable via a Davidson extension of BT |
| Repo and branch naming | still named for the abandoned streamer project |
