# Handoff — FPL audit remediation, post-review

**Review status:** Codex reviewed `98ada9f` and found 11 blockers (RB1–RB11
below). Claude verified every finding against the source — all stood — and is
fixing them in Codex's recommended order.

**Fix progress (2026-09-18):**

| Blocker | Status | Commit |
|---|---|---|
| RB4 same-GW Free Hit re-confirm corrupts base | **fixed** + regression test | re-confirming a free hit |
| RB5 Triple Captain formula / rule / prose; BB/TC values wired into advisor | **fixed** + tests replaced | triple captain passes to vice |
| RB6 Free Hit solve bought future captaincy | **fixed** + regression test | free hit valued on one week |
| RB7 override bypassed availability cap | **fixed** + invariant tests | team news blends before availability |
| RB8 `--confirm` marks the wrong forecast | in progress | |
| RB9 illegal/unknown chips accepted at confirm | not started | |
| RB1 snapshots not consumed by replay | not started | |
| RB2 replay forces horizon_gw=1 | not started | |
| RB3 replay does not carry captain/vice | not started | |
| RB10 first-snapshot-wins vs action-time | not started | |
| RB11 FH restoration needs the picks fetch | not started | |

Suite: **634 passed**.

**Branch:** `master` — 29 commits, `370de0e..98ada9f`

**Verification:** `python -m pytest -q` → **629 passed, 1 warning** in 75.35s

**Last reviewed:** 2026-09-17

This file supersedes the earlier completion claims. Read it together with:

- `docs/fpl-model-optimizer-audit-2026-09-17.md`
- `docs/superpowers/plans/2026-09-17-audit-p0-and-correctness.md`
- `docs/superpowers/plans/2026-09-17-audit-p1-honest-validation.md`

## Bottom line

Claude materially improved the repository:

- Chip inventory is dated and split into the two 2026/27 windows.
- Wildcard and Free Hit preserve banked free transfers correctly.
- Wildcard and Free Hit now build their own squads rather than decorating an
  ordinary transfer recommendation.
- Transfer enumeration covers every transfer count, including paid-hit plans.
- Calibration no longer invents points in blanks.
- Failed element-summary fetches are distinguishable from genuine newcomers.
- Forecasts are versioned and replay forecasts cannot replace live forecasts.
- A sequential manager-state replay now exists, and the old weekly rebuild is
  explicitly labelled as an oracle ceiling.
- Transfer selection now defaults to discounted horizon xP rather than allowing
  a one-week rank simulation to overrule the horizon.

However, the current replay is not yet evidence that the production policy works,
and a few chip/minutes paths can still produce wrong live advice. Fix the review
blockers below before continuing with B7/B8/R3.

## Review blockers

### RB1 — Critical: point-in-time snapshots are never consumed by the replay

**Where:** `scripts/run_walkforward.py:103-111`, `:132`, `:140-147`; compare
`fpl/data/snapshots.py:65-74`.

The script imports `snapshots`, but only calls `contamination_note()`. There is no
call to `snapshots.load()`. It normalizes one newest cached bootstrap and fixture
payload before the gameweek loop, then uses those current objects for every past
gameweek.

This means a future gameweek with a valid pre-deadline snapshot will make the
contamination banner disappear while the replay still uses today's prices,
status, news, team assignments, and fixtures. B13's snapshot half is therefore
infrastructure only, not wired behaviour.

**Fix:** For each replayed gameweek, load that gameweek's snapshot and build
`players`, `teams`, and `fixtures` from it. Fall back to current cache only when
the snapshot is absent or not point-in-time, and keep the contamination banner in
that case. Prefer versioned snapshots tied to the forecast manifest rather than a
single gameweek directory.

**Regression test:** Make current cache and the GW snapshot disagree on a
player's price/status/team and a fixture, run the replay builder, and assert the
snapshot values are used.

### RB2 — High: the “expected” replay is not the production transfer policy

**Where:** `scripts/run_walkforward.py:98-100`, `:141-147`;
`fpl/backtest/replay.py:127-133`.

The script forcibly sets `cfg.horizon_gw = 1`. The production transfer policy
chooses on discounted `xp_horizon` using the configured multi-gameweek horizon;
the replay's `xp_horizon` contains only the current week. It is therefore a
greedy one-week policy, not a replay of live decisions. The reported GW1-4 result
(`hold 236`, `expected 234`, `oracle 245`) is useful only as a smoke test and
must not be interpreted as evidence about the live optimizer.

**Fix:** Build the configured horizon at each historical decision point and score
only the current gameweek's realised points. If a one-week challenger is useful,
keep it under a separate policy name.

### RB3 — High: sequential replay does not carry the live captain/vice decision

**Where:** `fpl/backtest/replay.py:66-80`; `fpl/backtest/walkforward.py:118-136`;
compare `fpl/optimize/lineup.py:24-52`.

`Decision` has no captain or vice. `realised_score()` silently reselects the two
highest projected starters, while the live lineup uses `choose_captain()` and
accounts for the captain's non-appearance probability. The replay can therefore
score a different armband from the production recommendation.

**Fix:** Put captain and vice on `Decision`, populate them through the same live
lineup function, and have realised scoring honour those explicit IDs (including
vice takeover). Add a test where the live risk-aware captain differs from the
highest raw xP player.

### RB4 — High: confirming the same Free Hit twice destroys the permanent squad

**Where:** `fpl/cli.py:60-83`, `:188-203`; `run_gameweek.py:134-166`.

On a same-GW re-run, `resolve_current_squad()` returns the already-confirmed
temporary Free Hit squad. `run_gameweek.py` then passes that temporary squad back
as `base_squad`, and `record_transfers()` overwrites the saved permanent base.
The chip record is idempotent, but the base state is not.

This was reproduced directly: after two GW6 Free Hit confirmations,
`base_squad` became the temporary IDs `[101, 102, 103, ...]` rather than the
permanent `[1, 2, 3, ...]`.

**Fix:** If `state.freehit_event == gw` and a base already exists, preserve the
existing `base_*` fields regardless of caller input. Also expose the permanent
base through `LiveSquad` or avoid resupplying it on a same-event confirmation.
Add an end-to-end double-confirm regression test.

### RB5 — High: Triple Captain value encodes the wrong rule and double-discounts xP

**Where:** `fpl/optimize/actions.py:130-145`, `tests/test_actions.py:104-113`;
also the advice text at `fpl/optimize/chips.py:267-274`.

`xp_next1` is already unconditional expected points, so multiplying it by
`p_play` again discounts the captain twice. The code and test also assume the
chip is wasted when the captain does not play. The official FPL FAQ says the
**triple-points bonus passes to the vice-captain**.

Under the existing independence approximation, the chip's marginal value over
ordinary captaincy is:

`captain_xp + (1 - captain_p_play) * vice_xp`

The advice text also incorrectly says an early substitution wastes the chip.
Moreover, `triple_captain_value()` and `bench_boost_value()` are not called by
the pipeline; they currently validate only themselves.

**Fix:** Correct the formula and prose, replace the test that asserts the wrong
formula, and feed the real marginal value into chip timing. Official source:
https://www.premierleague.com/en/news/4661030

### RB6 — High: the Free Hit solve still includes captain value from future weeks

**Where:** `fpl/optimize/actions.py:87-103`;
`fpl/optimize/transfers.py:84-86`; `fpl/optimize/objective.py:108-151`.

`freehit_action()` passes `xp_next1` for the base XI term, but `add_captaincy()`
detects every `xp_gw*` column and adds captain bonuses for the whole projection
horizon. A one-week Free Hit can therefore buy a player partly because he is a
good captain after the temporary squad has expired.

**Fix:** Scope captain event columns to the current event for a Free Hit (or pass
an explicit event set into the shared objective). Test a player who is weak now
but has huge future captain projections and assert the Free Hit does not select
him for that reason.

### RB7 — High when an override is present: doubtful availability can be bypassed

**Where:** `fpl/model/minutes.py:226-252`.

Availability is applied to `p_start`, then a manual news override is blended
into that already-capped number. For a 25%-available player, a strong override
can lift `p_start` above 0.25. `p_play` is subsequently capped at availability,
leaving the invalid state `p_start > p_play`; the simulator uses `p_start` first
and can play the doubtful player far more often than the availability cap.

**Fix:** Blend the override into the fully-fit start probability, then apply the
availability multiplier once. Enforce and test the invariants
`0 <= p_start <= p_play <= availability <= 1` and `p_60 <= p_start`, including a
doubtful player with an override.

### RB8 — High validation risk: `--confirm` does not identify the forecast acted on

**Where:** `run_gameweek.py:117-121`, `:167-170`;
`fpl/backtest/manifest.py:119-140`, `:144-172`.

Every confirm command first creates a new forecast, then `mark_actioned()` marks
the newest version. That is not necessarily the earlier planning forecast the
manager used. If confirmation is run after the deadline, an explicitly actioned
post-deadline forecast wins unconditionally over every pre-deadline version and
can enter calibration.

**Fix:** Carry the forecast version ID from the planning report into confirmation
(for example `--forecast-version`). At minimum, default to the newest live
version strictly before the deadline and refuse to mark a post-deadline version
as actioned. Test planning → later confirmation and post-deadline confirmation.

### RB9 — Medium: confirmation accepts unknown or illegal chips

**Where:** `run_gameweek.py:56-62`, `:140-166`; `fpl/cli.py:173-203`.

`--applied-chip` has no choices and the confirmation path never calls
`chip_available()`. A typo or a second same-window chip can be persisted even
though the advisor and replay legality layers would reject it.

**Fix:** Restrict names to the four canonical chips plus `none`, validate against
the dated chip history and opening gameweek, and abort without mutating state on
an illegal use.

### RB10 — Medium: “first snapshot wins” is safe but not the action-time record

**Where:** `fpl/data/snapshots.py:16-24`, `:37-61`; `fpl/pipeline.py:437-444`.

The first pre-deadline run may be days before the deadline and before the
previous gameweek is finalized. Later pre-deadline team news is legitimate
information, not hindsight. Keeping only the first snapshot makes replay inputs
different from the actioned forecast even after RB1 is fixed.

**Fix:** Store immutable snapshot versions and select the snapshot referenced by
the actioned forecast, otherwise the newest snapshot strictly before deadline.

### RB11 — Medium resilience: Free Hit restoration unnecessarily depends on API picks

**Where:** `fpl/cli.py:47-56`, `:65-88`.

The previous-GW picks fetch happens before state is loaded. If that request fails
after a Free Hit, the function returns without using the locally saved permanent
squad even though it has everything needed to restore ownership and bank.

**Fix:** Load state first and allow the `base_*` restoration path before requiring
the temporary Free Hit picks endpoint.

## Status against the original audit

| Audit item | Reviewed status |
|---|---|
| B1 chip inventory/legality | Implemented. The GW19→GW20 Free Hit restriction is now independently verified by the official 2026/27 rules. |
| B2 FT carry over WC/FH | Implemented and verified. |
| B3 reversible Free Hit state | Partial: normal restoration exists; same-GW re-confirmation still corrupts the base (RB4). |
| B4 chip actions | Partial: WC/FH squads exist; TC value/rule is wrong and BB/TC value helpers are not integrated (RB5/RB6). |
| B5 transfer candidate breadth | Implemented. |
| B6 horizon vs rank reranking | Implemented for live transfers; `rank_transfers=false` by default. |
| B9 availability consistency | Partial: works without overrides; doubtful overrides bypass the cap (RB7). |
| B10 calibration fixture-count application | Implemented. |
| B11 DGW role evidence | Implemented. |
| B12 Tier 1 cutoff leakage | Implemented. |
| B13 point-in-time executable replay | **Not complete:** snapshots are not loaded, horizon is changed, and captain/vice is not carried (RB1-RB3). |
| B14 immutable/actioned ledger | Partial: versioning is good; actioned selection is not tied to the actual planning forecast (RB8). |
| B15 Tier 2 DNP/minutes leakage | Implemented as designed. |
| B16 failed fetch vs newcomer | Implemented as designed. |
| B17 inherited four-player club overage | Implemented. |
| R11 hidden Bench Boost floor | Implemented; default is explicitly `0.0`. |

The old GW1-4 sequential totals should remain labelled **contaminated smoke-test
output**, not “the first executable-policy measurement.” Until RB1-RB3 are fixed,
the policy being measured is not the production policy and a clean-snapshot
claim would be false.

## Still-open model work (after the blockers)

The earlier P2 list remains valid:

- **B8:** use per-event lineups or at least an exact one-week XI for the report.
- **B7:** align calibrated xP with simulation means and report the captain the
  selected objective actually scored.
- **R3:** simulate both sides of a fixture coherently so a goal and opposing clean
  sheet cannot coexist in the same scenario.
- **R2:** calibrate rival squads from real rank-cohort ownership/picks when data
  exists; meanwhile enforce generated-rival budget and club legality.
- **R1:** decide explicitly whether Mode 1 should use discounted expected points
  by default and leave threshold/rank utility as an opt-in chase mode.

Unchanged P3 items: R4 confidence propagation, R5 team/event minutes scenarios,
R6 cold starts and override staleness, R7 attack-strength double counting, R8
multi-period transfer/FT option value, R9 chip opportunity value beyond the
horizon, and R10 goalkeeper/bonus structure.

## Recommended implementation order for Claude Code

1. Add failing regression tests for RB4-RB7, then fix those live-decision paths.
2. Fix confirmation legality and exact forecast identity (RB8-RB9).
3. Wire snapshots into replay and restore the production horizon/captain decision
   (RB1-RB3), using versioned action-time snapshots (RB10).
4. Make Free Hit restoration work offline from `base_*` state (RB11).
5. Run the full suite and one end-to-end historical replay whose snapshot values
   deliberately differ from current cache.
6. Only then continue B8 → B7 → R3. Do not tune the model from the current
   sequential replay numbers.

## Tests that currently give false reassurance

- `tests/test_actions.py::test_triple_captain_is_worth_one_more_captain_return`
  asserts the incorrect TC formula and must be replaced, not preserved.
- `tests/test_snapshots.py` proves snapshot storage and the first-write policy,
  but no integration test proves the walk-forward script consumes a snapshot.
- `tests/test_replay.py` verifies many rules in isolation but does not verify the
  production horizon or production captain/vice decision.
- There is no same-GW double-confirm Free Hit test.
- There is no doubtful-player-plus-manual-override minutes invariant test.

No production source was changed during this review; only this handoff was
updated. The worktree was clean before the handoff edit.
