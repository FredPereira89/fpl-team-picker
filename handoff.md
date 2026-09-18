# Handoff — FPL audit remediation, post-review

**Review status:** five Codex reviews so far. Review 1 (RB1–RB11) fixed at
`ad025d1..ed0092c`; review 2 (RR1–RR7) fixed at `880fbc9..dfd6f10`; review 3
(three findings, below) fixed with the file's restoration; review 4 (four
findings) and review 5 (two cleanups) fixed after.

**Branch:** `master` — see `git log` for HEAD; every fix commit names its finding.

**Verification:** `python -m pytest -q` → **685 passed, 1 warning**.
`python scripts/run_walkforward.py --through 3 --no-save` runs end to end with
the pinned-snapshot, calibrated, horizon-start path.

**Last updated:** 2026-09-18 (Claude, after Codex re-review)

| Finding | Fix | Commit |
|---|---|---|
| RR1 pre-deadline `--confirm` marked its own rerun | the planning forecast is selected from the manifest BEFORE the confirm run writes; every run prints `Forecast version: <id>` to copy into `--forecast-version` | `880fbc9` |
| RR3 same-GW re-confirm could swap one chip for another | only the matching `(chip, event)` record is exempted for idempotence | `a50c131` |
| RR4 TC timing compared unlike currencies | timing uses the raw per-event approximations on both sides; the real marginal values appear only in the reason text | `0742db6` |
| RR2 actioned snapshot recorded but not consumed | `walkforward.actioned_snapshot()` pins the manifest's snapshot for both inputs and the contamination banner | `aaa2275` |
| RR5 calibration gated on `--no-save`, applied without decay | `walkforward.replay_calibration()` is independent of writing and passes `horizon_decay`; `--no-calibrate` opts out | `aaa2275` |
| RR7 synthetic start on `xp_next1` | starts on `xp_horizon` (production Mode 1) and is labelled a challenger; `--squad` for the manager's own season | `aaa2275` |
| RR6 snapshots omitted overrides/config | each capture stores `news.json` + `config.json`; the replay applies the overrides; element-summary reconstruction is classified explicitly via `SUMMARIES_NOTE`, printed by the script | `dfd6f10` |

### Third review (2026-09-18)

| Finding | Fix |
|---|---|
| 1. A legacy forecast (recorded before snapshots existed) replayed against a NEWER capture with no contamination warning, because `actioned_snapshot()` returned `None` for both "no forecast" and "forecast without snapshot" | `NO_SNAPSHOT` sentinel; `gameweek_inputs` forces a flagged fallback to the current cache; `contamination_note` lists the gameweek. Tests: `test_a_forecast_that_predates_snapshots_forces_a_flagged_fallback`, `test_no_forecast_at_all_still_allows_automatic_snapshot_selection`. |
| 2. `config.json` was archived but never applied — every historical week ran under today's settings | `gameweek_inputs` returns the archived config; `config_for_replay` applies its known fields; `replay_season`/`compare_policies` take `cfg_by_gw`; the script labels weeks without an archive a current-configuration challenger. Tests: `test_the_replay_returns_and_applies_the_archived_config`, `test_each_gameweek_is_replayed_under_its_own_config`. |
| 3. This file was internally inconsistent (status table, branch marker), and a scripted edit had corrupted it to 39 MB in `f71b7e0` | Restored from `96a1591` and corrected; the corrupt commit was dropped from history before any push. |

### Fourth review (2026-09-18)

| Finding | Fix |
|---|---|
| 1. The synthetic starting squad and bank used today's config | `replay.initial_state(xp, cfg, squad=None)` builds and banks under the FIRST week's archived config; tests show a budget change moving the squad and the bank. |
| 2. `rank_sims = 0` silently overrode an archived `rank_transfers=true` | Not implemented in the replay; such a week is now explicitly labelled a current-policy challenger in the output. |
| 3. Replay provenance hashed today's config | `save_predictions` receives `week_cfg`. |
| 4. Stale B4 row and replay paragraph | Corrected. |

### Fifth review (2026-09-18)

| Finding | Fix |
|---|---|
| 1. `config_for_replay` returned the shared fallback object, which the script then mutated | Always returns a copy; the rank-transfer note distinguishes an archived config from today's fallback. |
| 2. Stale header counts | Corrected. |
| 3. Claude misread the smoke run: 164/160 vs 101/100 were three- vs two-gameweek totals, not a config effect — GW1–3 have no archived config in this checkout | Acknowledged; no code change. |

Codex's re-review findings are kept below for the record.

## P2 progress (started 2026-09-18 after review 5)

| Item | Status | Where |
|---|---|---|
| B8 exact one-week XI for the report | **done** — `lineup.best_xi()` enumerates legal formations; `build_lineup(exact=True)` is the default and the replay's armband goes through it | `fpl/optimize/lineup.py` |
| B7a simulated means agree with calibrated xP | **done** — `simulate.moment_match()` scales each player's samples to `xp_next1` inside `pipeline._rank_context` | `fpl/model/simulate.py`, `fpl/pipeline.py` |
| B7b report the captain the rank layer scored | **done** — `pipeline._honour_rank_captain()`; `rank_stats["captain_reported"]` says whether it applied | `fpl/pipeline.py` |
| R3 coherent match scenarios | **done** — `simulate_event` runs per fixture; a side's goals are Poisson at the opponent's `xgc` and that draw IS the opponent's conceded count; goals allocated to on-pitch players with shares `w_i / max(Σw, λ)` (means preserved, or scaled to the team total when the player sum exceeds it — the R7 remedy). Assists and bonus still independent (documented residual). `simulate_event_detailed` exposes goals/conceded per scenario. | `fpl/model/simulate.py` |
| R2 rival field legality | **done (achievable part)** — `rank._repair()` makes every drawn XI ≤3 per club and priced under budget minus the cheapest legal bench (`_xi_ceiling`). Cohort-calibrated formation/captain shares still need pre-deadline public picks, which nothing in the repo fetches — that half stays open. | `fpl/optimize/rank.py` |
| R1 Mode 1 objective | needs a decision from the user (see "Still-open model work") | — |

## Re-review findings

### RR1 — High: a normal pre-deadline `--confirm` still marks its own rerun

**Where:** `run_gameweek.py:191`; `fpl/backtest/manifest.py:155-197`.

`--confirm` runs the complete pipeline first, creating a new live forecast, and
then calls `mark_actioned(deadline=...)`. If confirmation is before the deadline
(the normal case), that just-created forecast is eligible and is the newest, so
it is marked instead of the earlier planning forecast the manager actually
used. The deadline fix only distinguishes a **post-deadline** confirmation.
`--forecast-version` is an exact escape hatch, but it is optional and the report
does not surface a ready-to-copy version id.

Codex reproduced this with a planning version at September 10 and a confirmation
rerun at 13:00 on September 11, both before a 17:30 deadline:

`predeadline_confirm_marks = confirm-rerun`

**Fix:** make the forecast version explicit in the rendered recommendation and
require it on confirmation, or capture the candidate version before the
confirmation pipeline writes another one. Add the missing pre-deadline
planning → confirmation regression test.

### RR2 — High validation risk: the actioned forecast's snapshot link is recorded but ignored

**Where:** `scripts/run_walkforward.py:132,146`;
`fpl/backtest/walkforward.py:212-234`; `fpl/backtest/manifest.py:119-136`.

The manifest correctly stores `snapshot`, and `gameweek_inputs()` accepts a
`snapshot_version`, but the walk-forward caller passes neither the actioned
forecast's snapshot nor a deadline. It therefore selects the newest
pre-deadline capture, not the capture used by the actioned forecast. The
contamination banner makes the same unpinned selection.

Codex reproduced two pre-deadline snapshots: the actioned forecast referenced
the first (£5.5), while the replay silently read the later one (£5.7). Passing
the manifest's snapshot id returned the correct £5.5 input.

**Fix:** for each gameweek call `manifest.select_version()`, pass its
`snapshot` to both `gameweek_inputs(..., snapshot_version=...)` and
`contamination_note(..., versions_by_gw=...)`, and test the complete caller
path rather than the snapshot helper alone.

### RR3 — Medium: same-gameweek reconfirmation can replace one chip with another

**Where:** `run_gameweek.py:162-163`.

The legality check removes **all** chip events from the current gameweek before
calling `chip_blocked_reason()`. That correctly makes reconfirming the same chip
idempotent, but also hides a different chip already recorded in that gameweek.
For example, a recorded GW8 Bench Boost is removed and a GW8 Wildcard is then
accepted, violating the one-active-chip rule.

**Fix:** remove only the matching `(chip, event)` record for idempotence; retain
other same-event chips. Add a Bench Boost → Wildcard same-GW regression test.

### RR4 — Medium decision risk: current and future Triple Captain values use different currencies

**Where:** `fpl/optimize/chips.py:201-231`.

This week's value correctly includes vice takeover via
`triple_captain_value()`, but future weeks still use the best player's raw xP
from `captain_value_by_event()`. Comparing those directly biases the advisor
toward playing now. Codex constructed a case where this week was worth 10.0
including the vice and next week was worth 14.2 on the same formula; the advisor
reported that no better week was visible and recommended Triple Captain now
because it compared 10.0 against future raw xP of 9.5.

**Fix:** compute future armband marginal value on the same formula, with an
event-specific legal XI/captain/vice (B8), or conservatively compare raw captain
xP on both sides until that exists.

### RR5 — High validation risk once calibration activates: replay calibration is not production calibration

**Where:** `scripts/run_walkforward.py:159-162`.

The handoff recommends `--no-save`, but that flag also disables calibration,
even though production has `calibrate=true`. Without `--no-save`, the replay
calls `apply_calibration(xp, cal)` without the configured
`horizon_decay=0.85`, so the function's default `1.0` rebuilds an undiscounted
`xp_horizon`. This is dormant before five scored live gameweeks, then the
"expected" replay ceases to match the production transfer objective.

**Fix:** separate "write replay forecasts" from "apply available live
calibration", and always pass `decay=cfg.horizon_decay`. Add a calibrated
multi-week replay test with `--no-save` semantics.

### RR6 — Medium validation risk: snapshots still omit decision inputs

**Where:** `fpl/pipeline.py:411,440`; `scripts/run_walkforward.py:153`;
`fpl/data/snapshots.py`.

Production minutes use the loaded manual `news` overrides, but snapshots store
only bootstrap and fixtures and the replay calls `minutes_model()` without
`news`. Exact element-summary payload versions and the configuration payload are
also not archived (only collapsed source timestamps and a config hash are
recorded). The action-time forecast itself remains scoreable, but the replay
cannot yet reconstruct all inputs that produced it.

**Fix:** version the resolved overrides and config beside each snapshot, and
either archive exact element-summary inputs or explicitly classify their later
reconstruction as contamination.

### RR7 — Medium validation risk: the default replay starts from a one-week squad

**Where:** `scripts/run_walkforward.py:201-202`.

With no `--squad`, the initial squad is optimized on `xp_next1`; production Mode
1 builds on discounted `xp_horizon`. The script already admits that synthesizing
the initial squad is an advantage, but it is also a different objective. The
reported `hold`/`expected` totals therefore still do not represent the live
policy unless the actual starting squad is supplied.

**Fix:** require `--squad` for an executable-policy claim, or use
`xp_horizon` for the synthetic start and label that path as a challenger rather
than the manager's replay.

Codex changed no production source in its re-review. Claude's fixes are the six commits listed at the top.

| Blocker | Fix | Commit |
|---|---|---|
| RB4 same-GW Free Hit re-confirm corrupts base | first record of the base is final for that Free Hit | `ad025d1` |
| RB5 Triple Captain formula / rule / prose; BB/TC not wired | formula, vice takeover, prose and current-week wiring fixed; timing compares like with like (RR4) | `d85c0ae`, `0742db6` |
| RB6 Free Hit solve bought future captaincy | only the current event's column reaches the FH solve | `893cf27` |
| RB7 override bypassed availability | override blends into the fully-fit rate, availability applies once; invariants tested for every player | `01584a1` |
| RB8 `--confirm` marks the wrong forecast | post-deadline refused, explicit pinning, and the planning forecast captured before the confirm run writes (RR1) | `aa1644d`, `880fbc9` |
| RB9 illegal/unknown chips at confirm | unknown, repeated-window and same-GW-different chips all rejected (RR3) | `8645fc2`, `a50c131` |
| RB10 first-snapshot-wins | captures versioned, forecasts record `snapshot`, and the replay pins the actioned forecast's capture (RR2) | `f47db52`, `aaa2275` |
| RB1 snapshots never consumed | `walkforward.gameweek_inputs()` selects the point-in-time snapshot per gameweek, falls back to cache and flags it | `81ea874` |
| RB2 replay forced `horizon_gw=1` | configured horizon used; `expected_points_policy` documents that it needs the horizon frame; one-week policies strip to the current event via `one_week_frame()` | `81ea874` |
| RB3 replay reselected the armband | `Decision.captain/vice` set through `build_lineup`; `realised_score` honours them incl. vice takeover | `81ea874` |
| RB11 FH restoration needed the picks fetch | state loaded first; picks fetched only for an ordinary gameweek | `ed0092c` |

This file keeps Codex's original findings below for the record. Read it together with:

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
| B3 reversible Free Hit state | Implemented (RB4, RB11 fixed). |
| B4 chip actions | Implemented (RB5, RB6, RR4 fixed). |
| B5 transfer candidate breadth | Implemented. |
| B6 horizon vs rank reranking | Implemented for live transfers; `rank_transfers=false` by default. |
| B9 availability consistency | Implemented (RB7 fixed); invariants tested. |
| B10 calibration fixture-count application | Implemented. |
| B11 DGW role evidence | Implemented. |
| B12 Tier 1 cutoff leakage | Implemented. |
| B13 point-in-time executable replay | Implemented: pinned actioned snapshot, production calibration and decay, archived overrides and config, production start objective (RR2, RR5–RR7, third-review 1–2). GW1–4 remain irrecoverably contaminated and every replay says so. |
| B14 immutable/actioned ledger | Implemented: the planning forecast is captured before a confirmation writes (RR1); `--forecast-version` pins one explicitly; post-deadline versions refused. |
| B15 Tier 2 DNP/minutes leakage | Implemented as designed. |
| B16 failed fetch vs newcomer | Implemented as designed. |
| B17 inherited four-player club overage | Implemented. |
| R11 hidden Bench Boost floor | Implemented; default is explicitly `0.0`. |

The GW1–4 sequential totals remain **contaminated smoke-test output** — no
pre-deadline snapshot can exist for those gameweeks, and no archived config
either, so they run as a current-configuration challenger. For gameweeks that
DO have a snapshot, the replay is now the production policy: pinned capture,
archived overrides and config (including the starting squad and bank), the
configured horizon and decay, production calibration, and the live armband.
The one unreplayed production path is rank-decided transfers
(`rank_transfers=true`); such a week is labelled a current-policy challenger.
Still a handful of gameweeks; still not a verdict.

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

## Recommended implementation order (steps 1–5 done 2026-09-18)

1. ~~Add failing regression tests for RB4-RB7, then fix those live-decision paths.~~
2. ~~Fix confirmation legality and exact forecast identity (RB8-RB9).~~
3. ~~Wire snapshots into replay and restore the production horizon/captain decision
   (RB1-RB3), using versioned action-time snapshots (RB10).~~
4. ~~Make Free Hit restoration work offline from `base_*` state (RB11).~~
5. ~~Run the full suite and one end-to-end historical replay whose snapshot values
   deliberately differ from current cache.~~
6. ~~Close RR1–RR7.~~ Done at `880fbc9..dfd6f10`; third-review findings done after.
7. **Next:** continue P2 — B8 → B7 → R3. Do not tune the model from the current
   sequential replay numbers.

Two things to know before the next live run:

- `data/snapshots/` is **not** gitignored and will be written on every run
  (~1–2 MB per capture). Decide whether to track it; the replay only needs it
  locally.
- `--confirm` now refuses an illegal `--applied-chip` and refuses to mark a
  post-deadline forecast as acted on (it still records the squad and chip, and
  prints a warning).

## Test coverage after re-review

- TC formula test replaced (`test_triple_captain_is_worth_one_more_armband_return_and_passes_to_the_vice`).
- `tests/test_walkforward.py::test_a_point_in_time_snapshot_is_what_the_replay_reads`
  makes cache and snapshot disagree on price/status/club/fixture and asserts the
  snapshot wins.
- `tests/test_replay.py` now pins the production horizon
  (`test_the_expected_policy_sees_the_full_horizon`) and the risk-aware armband
  (`test_the_replay_scores_the_risk_aware_captain_not_the_top_projection`,
  `test_the_vice_takes_the_armband_when_the_captain_does_not_appear`).
- `tests/test_cli.py::test_reconfirming_a_free_hit_keeps_the_original_permanent_squad`.
- `tests/test_minutes.py::test_a_doubtful_player_with_a_strong_override_still_respects_availability`
  and `test_minutes_invariants_hold_for_every_player`.

The re-review's missing caller-level cases, each now with a named test:

- RR1: `tests/test_cli.py::test_a_pre_deadline_confirmation_marks_the_planning_forecast`,
  `::test_forecast_version_flag_pins_the_marked_version`.
- RR2: `tests/test_walkforward.py::test_the_replay_pins_the_actioned_forecasts_snapshot`.
- RR3: `tests/test_cli.py::test_a_different_chip_in_the_same_gameweek_is_refused`.
- RR4: `tests/test_chips.py::test_a_better_future_armband_week_still_holds_triple_captain`.
- RR5: `tests/test_walkforward.py::test_replay_calibration_uses_the_configured_decay`,
  `::test_replay_calibration_can_be_switched_off_independently`.
- RR6: `tests/test_snapshots.py::test_a_capture_keeps_the_overrides_and_config_the_run_used`,
  `tests/test_walkforward.py::test_the_replay_gets_the_overrides_the_live_run_applied`.
- RR7: script-level (`xp_horizon` start, labelled a challenger); verified by running
  `python scripts/run_walkforward.py --through 3 --no-save`.

Codex changed no production source in any of its reviews.
