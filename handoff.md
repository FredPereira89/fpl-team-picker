# Handoff — model improvement work (started 2026-08-27)

Pick this up cold. Everything needed to continue is in this file plus `docs/STATUS.md`.

## What this work is

A GW1 2026/27 out-of-sample audit found the picker's forecasting skill comes almost
entirely from the minutes model, not the scoring model. Full write-up with tables:
https://claude.ai/code/artifact/40440570-490d-407c-814e-537bc066bfe5

The audit's headline numbers (n=595 players, GW1 predictions rebuilt from the cached
pre-deadline snapshot, actuals from cached `element-summary`):

| Predictor | Spearman ρ vs actual GW1 points |
|---|---|
| Ownership % | 0.490 |
| FPL's own `ep_next` | 0.466 |
| `p_start` alone | 0.465 |
| **Our model, as shipped before this work** | **0.413** |
| Price | 0.303 |

`p_start` alone outranked the full expected-points model — the scoring layer was
making the ordering worse.

## Progress

| # | Item | State |
|---|---|---|
| P1 | Beta-binomial `p_start` (minutes model) | **done** |
| P2 | Clean sheets on a real goals scale | **done** |
| P8 | Prediction ledger + `score` command | **done** |
| P6 | FPL selling price in the transfer budget | **done** |
| P5a | Captain term in the MILP + horizon decay | **done** |
| P3 | Current-season form blending | **done** |
| P4 | Penalty-taker / set-piece signal | **done** |
| P7 | Rebuild bonus under the 2026/27 BPS table | **next** (was deferred; GW2 promoted it) |
| P5b | Full multi-period MILP (per-GW squad variables) | **deferred** |

Deferred items are deliberate, not forgotten — see "Deliberately not done" below.

### Measured effect so far (GW1, out of sample)

| | Before | After |
|---|---|---|
| Spearman, all players | 0.413 | **0.474** |
| MAE | 1.642 | 1.595 |
| Goalkeeper bias (pts/GW) | +0.700 | +0.599 |
| Forwards ρ | 0.282 | 0.412 |
| Midfielders ρ | 0.477 | 0.533 |
| Defenders ρ | 0.400 | 0.436 |
| Mean p_cs | 0.441 | ~0.35 (actual GW1: 0.300) |

Measured on GW1, so only P1, P2 and P4 can show up here: P3 needs current-season
gameweeks that did not exist before GW1, and P5a/P6/P8 change the optimizer and the
process, not the forecast. The model now also beats FPL's own `ep_next` (0.466) and
`p_start` alone (0.465) on this gameweek, neither of which it did before.

**One gameweek is one sample.** Re-run the check against GW2 before treating these
magnitudes as settled — the ledger (P8) now makes that a single command.

### GW2, scored 2026-08-31 — the gains hold, except for goalkeepers

| | GW1 (after) | GW2 |
|---|---|---|
| Spearman, all players | 0.474 | **0.595** |
| MAE | 1.595 | **1.35** |
| Bias, pts/player | — | **+0.04** |
| Goalkeeper bias | +0.599 | **+0.72** |
| Defenders ρ | 0.436 | 0.592 |
| Midfielders ρ | 0.533 | 0.634 |
| Forwards ρ | 0.412 | 0.563 |

614 players, 307 of whom appeared; ρ among those who appeared was +0.347, so the ordering
skill still leans on the minutes model. Scored from `event/2/live/` rather than 614
element-summary calls because the gameweek had only just ended — same join, same metrics.
Bonus was still provisional, so re-run `scripts/score_gameweek.py --gw 2` once GW2 is
`data_checked`.

**P1 and P2 are confirmed on a second sample. P7 is no longer optional.** Goalkeeper
over-prediction went +0.70 (pre-fix) → +0.60 (GW1) → **+0.72 (GW2)**: two gameweeks agree the
residual survives the clean-sheet fix, which leaves saves and bonus — exactly what P7 covers.
GK *rank* quality is fine either way (0.524, 0.529); it is the level that is wrong, and the
level is what the optimizer spends its budget against.

## What changed, file by file

**P1 — `fpl/model/minutes.py`**
`p_start` is now a beta-binomial on starts (`(starts + k·prior) / (38 + k)`) instead of
a minutes-weighted blend with k=900. New `start_priors()` estimates the positional prior
from players past `ESTABLISHED_MINUTES` (900) only — including the several hundred
never-played players dragged every starter down by ~0.10. New config key
`model.start_prior_games` (default 4.0), added to `fpl/config.py` and `config.yaml`.
Before: the pool-wide maximum `p_start` was 0.855 and the league summed to 208.7
starters against a true 220. After: 214.7, max 0.986.

**P2 — `fpl/model/strength.py`, `fpl/model/fixtures.py`, `fpl/pipeline.py`**
`att`/`dfn` are ratios centred on 1.0, and `p_cs = exp(-xgc)` was consuming them as if
they were goal counts, so the average clean sheet came out at 0.44 against a real rate
near 0.27. New `league_goals_per_team_match()` estimates the rate per 90 from the
baseline season (season totals ÷ 38 undercount, because no player features in all 38);
`team_fixture_frame()` takes a `league_gc` argument and the pipeline passes the estimate.
`att_mult` is deliberately left unscaled — it multiplies per-90 rates that already carry
their own units.

**P8 — `fpl/backtest/ledger.py`, `scripts/score_gameweek.py`, `fpl/pipeline.py`**
Every run now writes `data/predictions/gw{n}.parquet`. `scripts/score_gameweek.py --gw N`
joins a recorded forecast to real returns and reports rank quality, error and positional
bias, then saves the verdict to `data/predictions/last_score.txt`. `pipeline.trust_text()`
quotes that verdict when it exists, so the report stops repeating the proxy backtest's
goalkeeper claim once a real gameweek has been scored. The ledger is deliberately NOT
gitignored — it is the model's audit trail and is a few tens of KB per gameweek.

**P6 — `fpl/optimize/transfers.py`, `fpl/state.py`, `fpl/cli.py`, `fpl/pipeline.py`, `run_gameweek.py`**
New `selling_price(purchase, now)`: half the rise, in whole 0.1 steps, full loss on a fall.
`optimize_transfers` takes `selling_prices` and charges owned players at selling value on
both sides of the budget constraint, so holding a risen player stays free while the
proceeds from selling him are honest. Purchase prices persist in `data/state.json`
(`State.purchase_prices`) and are carried forward each week by `record_transfers`.
Players with no recorded purchase price fall back to market value and the CLI warns.
The mode-2 bank now credits selling proceeds rather than market value.

## How to verify

```
python -m pytest -q            # 232 passing as of P1, P2, P8, P6
python scripts/score_gameweek.py --gw 2 --no-refresh   # once GW2 is played
```

The out-of-sample check that produced every number above lives in the session scratchpad
(`gw1_eval.py`, `gw1_bench.py`, `gw1_fixes.py` under
`%LOCALAPPDATA%\Temp\claude\c--Users-user-Documents-FPL-Team-Picker\...\scratchpad\`).
If those are gone, they are straightforward to rebuild:

1. Load `data/cache/bootstrap-static_20260820T111206Z.json` and
   `data/cache/fixtures_20260820T111206Z.json` — the pre-GW1-deadline snapshot.
2. Load every `data/cache/element-summary-*_20260826*.json` for `history_past`
   (the baseline) and `history` (GW1 actuals, `round == 1`).
3. Run the pipeline's model steps — `normalize_players` → `apply_season_baseline` →
   `team_ratings` → `league_goals_per_team_match` → `team_fixture_frame` →
   `per90_rates` → `minutes_model` → `build_xp` with `from_event=1`.
4. Join `xp_next1` against actual `total_points` and report Spearman, MAE and bias
   by position.

No network access is needed — everything is cached.

**P5a — `fpl/model/xp.py`, `fpl/optimize/squad.py`, `fpl/optimize/transfers.py`, `fpl/pipeline.py`, `fpl/report/weekly.py`**
Both MILPs gained a captain variable (`Σ cap = 1`, `cap[i] ≤ start[i]`, objective `+ xp·cap`),
so the solver finally values a ceiling; `Squad.captain_id` reports who it picked. `build_xp`
now emits `xp_horizon` alongside `xp_next5`: the same points discounted by
`model.horizon_decay` (default 0.85) per gameweek. `xp_next5` stays the honest undiscounted
total the report shows a human; `xp_horizon` is what the optimizers maximise. The transfer
line in the report says the gain is discounted rather than letting it read as raw points.

**P3 — `fpl/data/normalize.py`, `fpl/model/scoring.py`, `fpl/model/minutes.py`, `fpl/pipeline.py`**
New `history_current_frame(summaries, before_event)` aggregates season-to-date totals from
`element-summary` history, strictly before the gameweek being predicted. New
`blended_rates()` finally calls the `blend_form` machinery that had been dead code.
`minutes_model` takes `current=` and counts this season's starts as ordinary binomial
evidence, using distinct rounds on record as the "chances to start" denominator — so a
mid-season signing is measured against his own club's games, not a full 38. For a player
with no Premier League history the PRICE is now the prior that current-season starts update,
rather than a fixed price guess that no evidence could move. That is the Tzolis blind spot closed.

**P4 — `fpl/data/normalize.py`, `fpl/model/scoring.py`**
`penalties_order`, `corners_and_indirect_freekicks_order` and `direct_freekicks_order` are
carried through normalization (nullable — a 0 fill would sort every non-taker ahead of the
real one) and `apply_set_piece_roles()` credits takers: +0.10 xG90 for penalties, +0.05 xA90
for corners, +0.02 xG90 for direct free kicks, at 25% for the second-choice taker. The
premium is scaled by `k / (minutes + k)` — the share of a player's rate that is still prior
rather than measured — so an established taker whose xG90 already contains last season's
penalties is not credited twice, while a new signing gets the full premium.

## Remaining work

Both remaining items are the ones deliberately left; see below. GW2 has now been scored and
the P1/P2 gains held, so that gate is cleared — and it promoted **P7 from optional to next**.
Take P7 first: the goalkeeper bias is measured, persistent across two gameweeks, and pointed
straight at the part of the model P7 rebuilds.

## Deliberately not done

- **P7 (BPS rebuild).** The 2026/27 BPS table changed — clearances/blocks/interceptions now
  score 1 BPS per 3 rather than per 2, the −1 for being tackled is gone, and goalkeeper
  saves were restructured. `expected_bonus` carries forward last season's realised
  `bonus90`, so it is biased for centre-backs (inflated), dribblers (deflated) and keepers
  (both ways). Fixing it properly means rebuilding expected BPS from component stats and
  modelling bonus as a probability of finishing top-3 in a match rather than a per-90
  average. That is a rewrite of `fpl/model/bps.py`, not a patch.
- **P5b (full multi-period MILP).** Per-gameweek squad/XI/transfer variables with free-transfer
  state, as in Çay's FPL-Optimization-Tools. Worth doing only once the projections underneath
  are worth planning five weeks against. P5a captures most of the value for a fraction of the work.

## House rules observed in this repo

- TDD: every change here went test-first, with the failure watched before implementing.
  Test count went 200 → 260.
- Docstrings explain *why*, not what; comments carry the reasoning behind non-obvious choices.
- Nothing is committed unless the user asks. As of this writing the P1/P2 changes are in the
  working tree, uncommitted.
