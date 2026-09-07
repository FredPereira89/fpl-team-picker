# FPL Team Picker — Codex Review

## Executive assessment

The repository has a strong foundation: a clear data → model → optimiser →
report pipeline, a read-only public FPL client, a sensible cache, and focused
tests. Its biggest limitation is not the use of MILP; it is that the current
forecast and optimiser do not yet represent the sequence of decisions that
determines an FPL season and, ultimately, rank.

The immediate goal should be trustworthy weekly decision support. Do not add
more optimisation complexity until state, data freshness, and the objective
are correct.

## Findings

### P0 — The five-gameweek objective fixes decisions that are free every week

`fpl.optimize.squad` and `fpl.optimize.transfers` select one starting XI and
one captain against an aggregate, discounted five-GW value. In reality, a
manager changes the XI, bench order, captain, and vice-captain for free each
gameweek.

This can distort player value: a player with one excellent fixture should be
started and captained only for that event, not treated as the captain for the
whole horizon. It also means the current optimiser cannot value rotation,
bench cover, doubles, and fixture swings accurately.

Recommended fix:

- Emit event-indexed predictions: `xP[player, gameweek]`, including scoring
  components and uncertainty.
- Optimise one lineup, captain, vice-captain, and bench order per gameweek.
- Model transfers, bank, selling prices, and free-transfer rollovers across
  the horizon.
- Add chip choices only after the baseline multi-period optimiser is working.

### P0 — Chip and free-transfer state can become incorrect

The pipeline calls `advise_chips(..., [])`, so it ignores the chips recorded in
`data/state.json`. A used chip can therefore be suggested again.

In addition, `reconcile()` derives free transfers only from the public
`event_transfers` history. It has no event-level record of a Wildcard or Free
Hit, which preserve free transfers. On the following gameweek a successful
history fetch can overwrite the correct locally tracked balance with an
incorrect value.

`--confirm` also records the number of recommended transfers and purchase
prices but keeps the old confirmed squad and bank. A rerun before the deadline
can therefore plan from the pre-transfer squad.

Recommended fix:

- Carry `chips_used` from local state into `LiveSquad` and `pipeline.run`.
- Store chip use by gameweek, not merely as an unordered list.
- Persist the exact applied squad, bank, transfers, and chip after a manual
  confirmation.
- Make confirmation explicit: require the applied transfer IDs and chip, or a
  signed recommendation identifier, rather than assuming every recommended
  move was made.
- Add integration tests for reruns after transfers, Wildcard, and Free Hit.

### P1 — Bench weights are configured but not honoured

The configuration defines four bench-slot weights, but both optimisers replace
them with their mean. This makes Bench 1 as valuable as Bench 3 and ignores
the actual auto-substitution order.

Recommended fix:

Use an auto-substitution scenario model. It should depend on each starter's
probability of a zero-minute appearance, bench order, goalkeeper substitution
rules, and valid-formation rules. At minimum, optimise explicit bench slots
rather than averaging their weights.

### P1 — Several model and freshness settings are silently inactive

The following fields are loaded but are not used in the weekly decision path:

- `model.form_half_life_gw`
- `news.max_age_hours`
- `data.cache_ttl_matchday_hours`
- `odds.provider`

`ew_mean()` exists but is not called, so current-season “form” is an aggregate
of all recorded rounds, not a recency-weighted estimate. The odds-aware team
ratings method exists but the pipeline never supplies a provider.

Recommended fix:

- Either wire each configuration value through to observable behaviour and add
  behavioural tests, or remove it until implemented.
- Date team-news overrides structurally and reject or flag stale ones.
- Apply the shorter cache TTL on matchdays.
- Add an explicit odds-provider interface and provenance to every forecast.

### P1 — Fixture strength remains largely last-season strength

Individual player rates use season-to-date data, but team attack and defence
ratings are computed from the completed-season baseline before current-season
data is used. This underreacts to new managers, promoted teams, major player
moves, injuries, and genuinely changing team strength.

Recommended fix:

Build an updating, shrunk team attack/defence model. A useful first version is
a home/away hierarchical Poisson model using current-season xG/xGA, shrunk to
last season and supplemented by betting-market goal or clean-sheet odds.

### P1 — Minutes and several FPL scoring rules are oversimplified

The minutes model uses fixed values for minutes when starting and appearing as
a substitute. It equates the chance of 60 minutes with the chance of starting.
Those assumptions directly affect appearance points, clean sheets, defensive
contributions, autosubs, and captaincy.

Keeper scoring is especially likely to be distorted. Saves and conceded-goal
deductions are calculated as continuous expectations even though FPL awards
them in thresholds. Bonus is carried forward as historical `bonus90`, rather
than being estimated from projected BPS. This is consistent with the observed
goalkeeper bias in the production score ledger.

Recommended fix:

- Estimate separate probabilities for start, any appearance, and 60 minutes;
  fit them by player, team, and recent selection history.
- Use discrete outcome distributions for save points and conceded-goal
  deductions.
- Rebuild expected bonus from projected BPS components and validate it
  separately by position.
- Treat manual team-news overrides as documented, dated evidence rather than
  a permanent hidden model input.

### P1 — Expected points alone is not a complete rank objective

The project contains ownership data, but its risk/ownership objective is not
implemented. Maximising expected points is the right baseline, particularly
early in the season, but it is insufficient to maximise expected rank later
on.

Recommended fix:

After calibrating the xP engine, simulate player outcomes and compare the
candidate squad to an ownership-weighted manager field. Let the user choose a
rank context and risk preference: expected points for safety, or a controlled
rank-utility objective when chasing.

### P1 — Validation is promising but insufficiently reproducible

The production forecast ledger overwrites a prediction file per gameweek, so
it loses earlier pre-deadline forecast versions. Scores can also be based on
provisional returns. Two gameweeks are useful diagnostics, not enough evidence
to declare a model component reliable.

Recommended fix:

- Store immutable forecast versions with creation time, source snapshot IDs,
  model/config version, and prediction components.
- Score only data-checked gameweeks, or label provisional scores and
  re-score automatically once finalized.
- Run rolling-origin backtests with no look-ahead leakage.
- Measure start/60-minute calibration, clean-sheet calibration, xP MAE,
  transfer value after hits, captain performance, and full-team score against
  FPL-xP and template baselines.

## Recommended implementation order

1. **State and safety:** repair chip/FT reconciliation, persist confirmed
   squad and bank, activate cache/news freshness checks, and add integration
   tests.
2. **Forecast observability:** version all forecasts and inputs; report xP
   components, uncertainty, and source freshness.
3. **Core forecast quality:** improve minutes, discrete goalkeeper scoring,
   BPS, current team strength, and recency-weighted form.
4. **Multi-period optimisation:** choose transfers, weekly lineups, captaincy,
   bench order, and chip timing jointly over the horizon.
5. **Rank-aware strategy:** add ownership-weighted simulations and a
   user-selectable risk/rank objective.

## Existing strengths

- Strong separation between data collection, modelling, optimisation, and
  reporting.
- Read-only public FPL API access and a cautious stale-cache fallback.
- Correct attention to squad constraints, selling value, and transfer hits.
- Clear user-facing reports that distinguish recommendations from actions.
- Focused unit coverage, including several regressions from prior live use.

## Notes on the existing Gemini review

Its bench-order and goalkeeper observations are useful. Vice-captain modelling
is also worthwhile, but it should use the probability the captain does not
appear at all, rather than simply `1 - p_start`.

Async player-history fetching and combining the transfer-count loop into one
MILP are lower priorities. The client deliberately rate-limits requests, so
asynchrony does not solve the important data problem; the major optimisation
gap is the fixed-XI/fixed-captain aggregate horizon.
