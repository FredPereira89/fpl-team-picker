# Handoff — FPL audit remediation, post-review

**Status (2026-09-18):** Review 6's six findings (C6-1..C6-6) and both findings
from its first re-review are fixed. A final consistency pass found three related
rank-simulation gaps — vice-captain inheritance, legal autosubs, and decision-
consistent gain probabilities/metadata — and fixed those as well. Reviews 1–7
are closed. Review 1 (RB1–RB11)
fixed at `ad025d1..ed0092c`; review 2 (RR1–RR7) fixed at `880fbc9..dfd6f10`;
review 3 (three findings) fixed with the file's restoration; review 4 (four
findings) and review 5 (two cleanups) fixed after. **R10 (bonus from ranked
simulated BPS) is implemented and has completed six review passes** — the
simulation-side scorer and its 2026/27 BPS coefficients are validated
against real GW1-4 fixtures with leave-one-gameweek-out cross-validation; a
`build_xp`/Mode 1 calibration attempt was tried, checked against real data,
found to cut true awarded bonus by more than half, and reverted (see its
design section below — `xp_next1`'s bonus term is intentionally back to a
plain `bonus90` projection). **R8 (rolling multi-period transfer MILP) is now
implemented as an opt-in live policy and a default walk-forward challenger**;
its terminal FT value still needs point-in-time replay evidence before the
new policy should become the live default. R8's Codex-found ownership-tilt
findings (below, two rounds) are fixed and regression-tested. The rest of R8
was committed alongside the first-round fix at `b800803`. Keep the live
switch off pending replay evidence for the terminal FT value.

### R8 Codex review (2026-09-18/19) - two rounds, both fixed

| Round | Severity | Status | Finding / next action |
|---|---|---|---|
| 1 | Medium | **fixed** | `multiperiod._solve_path()` maximises ownership-tilted event projections, then reconstructs `objective_xp` in raw forecast units. `optimize_multi_period()` calculated raw gain against the forced-wait path but returned the tilted winner even when that raw gain was negative, and `pipeline._choose_transfers()` unconditionally selects the rolling plan when the feature is enabled. Direct reproduction with the supported `risk_profile="differential"`, `ownership_weight=1.0` configuration replaced a 10.0-xP bench player with an 8.0-xP one and returned a negative raw gain that would have been reported as a suggested net gain. |
| 2 | Medium | **fixed** | Round 1's guard compared `chosen` against a `wait` baseline solved with the SAME tilted `cfg` — so `wait`'s own post-hold weeks could also make a tilt-driven, raw-negative swap, silently lowering the bar `chosen` had to clear. Reviewer reproduced a deterministic case where round 1's fix still reported `gain=+3.203` while the true disadvantage against a genuinely raw-optimal wait path was `-2.543` (returned plan 412.497 raw xP vs. best raw-xP wait 415.040). |

Why the one-period solver never has this bug at all: its wait/hold plan is
just another candidate in the pool `_choose_transfers` maximises on raw
`net_xp` (`fpl/optimize/transfers.py`'s `_attribute_gain`), so a
tilt-favoured pick that turns out raw-negative can never win — the hold plan
simply scores higher and is chosen instead. The rolling solver solves
`chosen` and `wait` as two independent MILP solves with no shared pool, so
that guarantee had to be made explicit in two steps:

- **Round 1**: `optimize_multi_period()` falls back to the `wait` baseline
  whenever `chosen.gain < 0`, instead of unconditionally returning the
  tilted winner.
- **Round 2**: the `wait` baseline itself is now solved with
  `dataclasses.replace(cfg, ownership_weight=0.0)` rather than `cfg`
  unchanged. With tilt off, `tilted_frame` is the identity (its early exit
  when every factor is 1.0), so that solve directly maximises the same
  quantity reported as raw xp — the true ceiling of what holding this week
  can reach, which always dominates (never trails) a same-cfg tilted wait
  solve. `chosen.gain` is now judged against that true ceiling, closing the
  gap the reviewer found.

Once `wait` is raw-optimal, `pipeline._choose_transfers()`'s unconditional
selection of the rolling plan (`fpl/pipeline.py` around line 490) is no
longer a separate concern: the plan it selects is raw-gain-safe by
construction, so no additional pipeline-level guard was needed.

Regressions in `tests/test_multiperiod.py`, each verified to fail against
the code they target and pass after their fix (`git stash` could not isolate
the new, still-uncommitted-at-the-time file, so both were checked via a
manual copy/revert/retest instead):
- `test_ownership_tilt_never_recommends_a_raw_negative_transfer` (round 1).
- `test_wait_baseline_is_judged_on_a_raw_optimal_hold_not_a_tilted_one`
  (round 2) — also asserts the same-cfg tilted wait solve for this fixture
  really does make the bad swap (`tilted_wait.weeks[1].out_ids == [victim]`),
  so the test documents the mechanism, not just the outcome.

Verification: round 1 pre-fix baseline **764 passed**; post-round-1 **765
passed**; post-round-2 **766 passed**, 1 pre-existing warning throughout.
`compileall` and `git diff --check` passed on both reviews. Round 1 committed
at `b800803` (with the rest of R8); round 2's fix is committed separately
below. `fpl/model/bps.py`, `fpl/model/simulate.py`, `scripts/validate_bps.py`
and their tests remain uncommitted and unreviewed here — unrelated to R8,
need their own look before landing.

### Review 6 fix progress (2026-09-18)

| ID | Status | Fix | Commit |
|---|---|---|---|
| C6-1 | **fixed** | `_with_armband` now copies `lineup.xi` back into the Decision, not only captain/vice, so `step()` scores the exact XI the armband was chosen from. Regression: `tests/test_replay.py::test_with_armband_copies_the_exact_xi_not_only_the_captain`, `::test_a_policy_using_with_armband_scores_the_exact_weekly_xi`. | (see git log: "replay scores the exact weekly XI") |
| C6-2 | **fixed** | New `pipeline._with_weekly_xi()` substitutes the exact one-week XI (`best_xi`) into every candidate's `starting_ids` BEFORE `pick_best_squad`/`_p_gain_positive` run, in both `_choose_squad` and `_choose_transfers`. `_honour_rank_captain` recomputes `Lineup.xp` when it overrides the captain. Regression: `tests/test_pipeline.py::test_rank_scoring_and_captaincy_use_the_exact_weekly_xi_not_the_horizon_one` (an engineered horizon/weekly-XI swap — player benched on the horizon, dominant this week — reproduced the bug end to end: `mean_points` 32.0→130.0, `captain` 1→13 once fixed), `::test_honour_rank_captain_recomputes_lineup_xp_when_captain_changes`. | (see git log: "rank scoring uses the exact weekly XI") |
| C6-3 | **fixed** | `weekly.py` now branches the "Against the field" selection sentence and the "No transfer recommended" hold message on `rank_stats["decided_by"]`, and renders a caveat when `captain_reported is False` instead of only recording it. Regression: `tests/test_report.py::test_the_field_section_says_expected_points_decided_when_they_did`, `::test_the_hold_message_names_expected_points_when_they_decided`, `::test_a_rejected_rank_captain_is_disclosed_not_hidden`. | (see git log: "rank scoring uses the exact weekly XI") |
| C6-4 | **fixed** | `actuals_from_summaries` now returns `fixture_count` (rows per round, from the existing groupby); `probability_scores` restricts scoring to `fixture_count == 1`, treating a missing column as 1 for backward compatibility. Verified empirically that blanks were already excluded by `score_gameweek`'s inner merge (no history row exists for a round with no fixture) — only the double-gameweek half of the finding reproduced; the commit documents that distinction rather than implementing the blanks fix Codex suggested for a bug that wasn't there. Regression: `tests/test_ledger.py::test_actuals_from_summaries_reports_the_fixture_count`, `::test_probability_scores_exclude_double_gameweek_rows`, `::test_probability_scores_without_a_fixture_count_column_score_everyone`, `::test_score_gameweek_carries_fixture_count_through_to_probability_scores`. | (see git log: "probability scoring excludes double gameweeks it cannot honestly settle") |
| C6-5 | **fixed** | Verified numerically that a single binary appearance event's marginal is Bernoulli(mean) for ANY generating mechanism with that mean — a per-scenario Beta draw provably cannot "widen" it, so that framing in the earlier R4 commit was an overclaim of my own. The REAL, confirmed bug was `p60_given_start = p_60/theta` using the RANDOM draw as denominator, which strictly lowers `E[reached_60]` below the intended `p_60` once clipping fires (reproduced numerically: buggy 0.402 vs intended 0.45). Fix: removed the Beta draw entirely rather than patch around ineffective machinery that had already caused a regression — `p_start`/`p_60` now stay at their fixed model values throughout `_on_pitch()`, so `p60_given_start` uses the fixed denominator. `start_evidence` stays computed/exposed as a labelled, currently-unused confidence signal; R4 (epistemic spread) is honestly still open and needs a PERSISTENT theta reused across a multi-week decision. R8 now supplies the rolling decision structure, but those posterior worlds are not yet built into it. Regression: `tests/test_simulate.py::test_a_thin_evidence_flag_on_the_minutes_frame_does_not_move_p_60` (verified fails pre-fix — 3.598 vs 3.766 expected — via `git stash` on the source files alone, passes post-fix), `::test_p60_given_start_uses_the_fixed_p_start_not_a_random_draw`, `::test_an_unavailable_player_never_appears`, `::test_start_evidence_does_not_change_a_single_gameweeks_simulation`. | (see git log: "remove the per-scenario Beta draw that biased the 60-minute marginal") |
| C6-6 | **fixed** | Reproduced numerically first: a pool where the pool-wide `_xi_ceiling()` scalar (cheapest keeper + 3 cheapest outfielders anywhere in the pool) passes a 3-5-2 XI costing 78.8, but that XI starts all 5 MIDs including the pool's 3 cheapest outfielders, so its TRUE cheapest legal bench (DEF/FWD, since MID has 0 bench slots left) costs 25.3 — a real total of 104.1 against a 100.0 budget, genuinely infeasible despite passing the old check. Fix: new `_cheapest_legal_bench()` computes, per position, exactly `SQUAD_SPLIT - in_xi` more players needed, cheapest first, skipping anyone already in the XI or whose club is already at `MAX_PER_CLUB`; `_repair()`'s stopping condition uses this exact check instead of the scalar (the swap heuristic inside a pass stays a cheap approximation, since the exact check re-verifies on every pass). `by_pos_sorted` is precomputed once per `sample_rival_squads` call, sorted by price ascending, since the check can run up to `REPAIR_PASSES` times per rival. `_xi_ceiling()` is removed — no callers remained. Regression: `tests/test_rank.py::test_a_ceiling_passing_xi_that_cannot_be_completed_is_repaired` (pins the exact numeric scenario above), `::test_no_rival_xi_costs_more_than_a_legal_fifteen_allows` (rewritten to assert genuine completability via `_cheapest_legal_bench` on every sampled rival XI, using a club-diverse pool rather than the pre-existing `_crowded_pool` fixture, which puts every GKP on one club and can make some XIs genuinely uncompletable regardless of price — a club-cap pathology `_repair` already documents it may not resolve, not a price-ceiling bug). Verified both new tests fail (ImportError) against pre-fix code via `git stash` on `fpl/optimize/rank.py` alone. | (see git log: "rival XI completability is checked exactly, not by a pool-wide scalar") |

Suite at time of writing this checkpoint: **723 passed, 1 warning** — all six
review-6 findings (C6-1..C6-6) are now fixed, tested, and committed.

### Review 6 re-review (2026-09-18) — Codex checked the C6 fixes, found two more, both now fixed

| Finding | Fix |
|---|---|
| High — C6-2 not fully closed: `score_candidate()` always re-optimises the armband per candidate, so when expected-points mode kept `build_lineup`'s own captain (or a rank captain got rejected for sitting outside the exact XI), `mean_points`/`p_beat_target`/`rank_percentile` still described RANK'S captain, not the one reported. Codex's own repro: rank captain 1 mean 29.0, reported captain 2 actually scores 28.0. A second path in the same finding: a Wildcard/Free Hit can replace the entire squad AFTER rank stats were computed for the ordinary plan, leaving stale numbers attached to a squad that no longer exists. | `score_candidate()` takes an optional fixed `captain` instead of always optimising one. `_choose_squad`/`_choose_transfers` stash the samples/rival_scores/bar/target/starting_ids/penalty each candidate was scored against as a private `_ctx` on `rank_stats`. `_honour_rank_captain()` re-scores all four stats for whichever captain ends up reported (honoured or not) using that context, then strips `_ctx`. A chip override now sets `rank_stats = None` instead of leaving it attached to a squad it never describes. Regression: `tests/test_pipeline.py::test_honour_rank_captain_rescores_stats_for_the_final_captain` (pins the audit's 29.0/28.0 exactly; verified fails pre-fix via `git stash`), `::test_honour_rank_captain_rescores_even_when_the_rank_captain_is_honoured`, `::test_a_wildcard_chip_clears_the_stale_rank_stats`. Commit: "rank stats describe the final captain; chip overrides clear stale stats". |
| High — C6-6's bench check was still greedy: `_cheapest_legal_bench()` filled each position independently, cheapest-first, with no backtracking across positions — so taking the cheapest reserve keeper from a club already near the cap could use up the only room a later position's one remaining legal candidate needed, wrongly reporting a real bench as impossible. | Replaced the per-position greedy fill with an exact depth-first search over every still-needed bench SLOT (not per position), backtracking on a club-cap conflict. Two bugs found and fixed ALONGSIDE this while stress-testing: (1) `_repair`'s own swap-candidate pool never excluded the player being removed, so a "swap" could silently re-pick itself and never change anything; (2) the search's own pruning bound (cost-so-far vs. best full solution) gave no benefit once many candidates tied on price — a routine case, not a synthetic one — so a partial path never got pruned until full depth, measured at over a second per call on a ~180-player pool; fixed with a per-slot cheapest-remaining lower bound added to the running cost, restoring sub-millisecond calls. `_repair` also now detects when its fast heuristic is about to repeat a state (a genuine 2-cycle between two players with no other legal partner, reproduced directly: the picked XI never changed across 60 traced passes) and falls back to a randomly chosen `out` instead of the deterministic one. Regression: `tests/test_rank.py::test_cheapest_legal_bench_backtracks_past_a_blocking_first_choice`, `::test_cheapest_legal_bench_stays_fast_with_many_tied_prices`, `::test_repair_escapes_a_two_player_deadlock` — all three verified to fail against pre-fix code via `git stash`. Full suite (20,000 sampled rival XIs across 50 pools with a realistic 20-club spread): 0 illegal. Commit: same as above (both findings landed together). |

Suite after the re-review fixes: **729 passed, 1 warning**.

### Review 7 implementation (2026-09-18) — Codex closed the remaining rank-simulation gaps

| Finding | Implementation |
|---|---|
| Rank scenario scores omitted vice-captain inheritance and legal autosubs, so simulated rank metrics could differ from the reported lineup under DNP scenarios. | `simulate_event_detailed()` now exposes an explicit per-player appearance mask, separate from points. `rank.lineup_scores()` uses that mask to apply captain-to-vice inheritance and exact FPL autosubs: keeper for keeper, bench order respected, and only formation-legal outfield substitutions. `best_armband_by_rank()` chooses the captain/vice pair using those same scenario rules. |
| `_p_gain_positive()` independently selected sample-mean captains instead of comparing the captain/vice decisions shown to the user. | The gain diagnostic now scores each candidate's final production `Lineup` with the same appearance-aware scorer. It therefore compares the reported XI, bench order, captain and vice rather than synthetic armbands. |
| Re-scored rank metadata could retain the rank layer's old captain even when the final reported captain differed. | `_honour_rank_captain()` now re-scores the final captain/vice pair, exposes that pair as `rank_stats["captain"]`/`["vice"]`, and retains the rank layer's suggestion separately as `rank_preferred_captain`/`rank_preferred_vice`. |

Regression coverage includes zero-point appearances (which must retain the
armband), captain DNP/vice takeover, formation-constrained bench order, final
rank metadata, and a gain-probability case where independently re-optimising
the armband would give the wrong answer.

Suite after the review-7 implementation: **733 passed, 1 warning**.

**Branch:** `master` — see `git log` for HEAD; every fix commit names its finding.

**Verification:** `python -m pytest -q` → **733 passed, 1 warning**, after all
review fixes. Focused review probes reproduced the replay-XI mismatch, stale
captain xP, double-GW probability mis-scoring, the Beta/p60 mean shift, and
the rival-XI completability gap described in review 6 — each pinned by a
regression test that fails against pre-fix code (via `git stash` on the
source file alone) and passes post-fix. Claude's prior
`python scripts/run_walkforward.py --through 3 --no-save` smoke run completed
end to end with the pinned-snapshot, calibrated, horizon-start path; Codex
did not repeat that network/cache-dependent run in review 6.

The warning is the existing SciPy `ConstantInputWarning` in
`tests/test_backtest_aggregate.py`; it is unrelated to these changes.

**Last updated:** 2026-09-18 (Codex R8 review; one finding open)

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

### Sixth review (2026-09-18) — historical decision, all findings resolved

**Decision at the time: request changes.** This section is retained as the audit
trail; the current status and review-7 section above supersede it.

| ID | Severity | Original finding | Required correction |
|---|---|---|---|
| C6-1 | **High — replay validity** | `replay._with_armband()` calls the exact weekly lineup builder but copies back only captain/vice. `step()` still scores `decision.starting_ids`, the old horizon XI. A focused probe produced a captain who was not in the XI being scored. | Copy `lineup.xi` into the decision as well as the armband. Add a caller-level replay test where the weekly and horizon XIs differ and assert both the scored XI and captain. |
| C6-2 | **High — rank decision/report validity** | `pick_best_squad()`, `_p_gain_positive()` and the saved rank statistics score every candidate's horizon `starting_ids`; the pipeline re-picks the exact weekly XI only afterwards. Therefore default rank diagnostics describe a different XI, and opt-in `rank_squad`/`rank_transfers` can decide on a lineup that is not fielded. `_honour_rank_captain()` can reject the rank captain but leaves the old simulated statistics in place; when it accepts a different captain it also leaves `Lineup.xp` stale. | Derive the exact one-week XI for every candidate before all rank scoring and gain diagnostics. Recompute rank statistics and lineup xP from the same XI/captain ultimately reported. Add integration tests with a horizon/weekly XI swap. |
| C6-3 | **Medium — user-facing correctness** | The weekly report ignores `rank_stats["decided_by"]`. It always says the squad was chosen by rank and, for a hold, says no plan beat the field—even under the new default where expected horizon points decided. | Branch both explanations on `decided_by`; state that rank is diagnostic when expected points decided. Render or remove `captain_reported` rather than storing an invisible caveat. |
| C6-4 | **Medium — validation correctness** | Stored `p_play`/`p_60` are per-fixture probabilities, while `probability_scores()` compares them with gameweek-total minutes. A blank therefore scores a valid 0% event opportunity as a failed ~90% appearance forecast; doubles also need event aggregation, and summed minutes cannot establish whether a player reached 60 in either fixture. | Store event-level probability forecasts using fixture counts, or score only single-fixture player-gameweeks. Preserve per-fixture actual minutes if `p_60` is to be scored for doubles. Add blank and DGW tests. |
| C6-5 | **Medium — R4 not achieved** | Drawing a fresh Beta `p_start` immediately before one Bernoulli appearance draw does not widen the one-event posterior-predictive appearance distribution: it collapses to Bernoulli at the same mean. Worse, `p_60 / drawn_p_start` with clipping lowers the 60-minute marginal (focused probe: modeled 0.400, simulated 0.320). Moment matching hides the points-mean movement but not the distorted event distribution. | Keep the conditional 60-minute probability fixed from the base model. Represent epistemic uncertainty at a persistent parameter/world level (especially across a multi-week decision), or describe R4 as still open; test appearance and 60-minute marginals directly, not only total-points mean/variance. |
| C6-6 | **Medium — R2 only partial** | `_xi_ceiling()` subtracts the cheapest keeper plus any three cheapest outfielders, regardless of the XI formation, whether those players are already in the XI, the missing squad positions, or the club cap. Passing that ceiling does not prove the XI can be completed to a legal 15. Example: the generic ceiling was £84m while a 3-5-2 XI in the focused pool had a true £76m ceiling. | Construct a cheapest legal complementary bench for each sampled XI (positions, distinct players, club cap), or sample a legal 15 first and then choose its XI. Assert full-squad completion in tests. |

Codex's re-review findings are kept below for the record.

## P2 progress (started 2026-09-18 after review 5)

| Item | Status | Where |
|---|---|---|
| B8 exact one-week XI for the report | **done (C6-1/C6-2 fixed)**. `lineup.best_xi()` enumerates legal formations; `replay._with_armband()` now copies the exact weekly XI back into the Decision it scores; `pipeline._with_weekly_xi()` substitutes the exact one-week XI into every candidate before rank scoring runs. | `fpl/optimize/lineup.py`, `fpl/backtest/replay.py`, `fpl/pipeline.py` |
| B7a simulated means agree with calibrated xP | **done** — `simulate.moment_match()` scales each player's samples to `xp_next1` inside `pipeline._rank_context` | `fpl/model/simulate.py`, `fpl/pipeline.py` |
| B7b report the captain the rank layer scored | **done (C6-2/C6-3/review 7 fixed)**. Rank evaluates captain/vice pairs with vice inheritance and legal autosubs on the exact weekly XI. `_honour_rank_captain()` re-scores whichever final pair is reported; `rank_stats["captain"]`/`["vice"]` identify that pair, while `rank_preferred_*` preserves the rank layer's suggestion. | `fpl/optimize/rank.py`, `fpl/pipeline.py`, `fpl/report/weekly.py` |
| R3 coherent match scenarios | **done** — `simulate_event` runs per fixture; a side's goals are Poisson at the opponent's `xgc` and that draw IS the opponent's conceded count; goals allocated to on-pitch players with shares `w_i / max(Σw, λ)` (means preserved, or scaled to the team total when the player sum exceeds it — the R7 remedy). Bonus is now match-ranked, not independent (R10). Assists remain independent (documented residual). `simulate_event_detailed` exposes goals/conceded/bonus per scenario. | `fpl/model/simulate.py` |
| R2 rival field legality | **done (C6-6 fixed)**. `rank._repair()` enforces the XI club cap and an exact per-XI `_cheapest_legal_bench()` check, which proves a distinct, position-correct, club-legal bench can complete the squad within budget — not merely a coarse pool-wide price ceiling. Cohort calibration remains open. | `fpl/optimize/rank.py` |
| R1 Mode 1 objective | **done** — expected points decide the Mode 1 squad; rank reports (`rank_stats["decided_by"]`, `rank_would_choose`); `optimizer.rank_squad` opts back in. Ties on the bar are worth half. **Behaviour change for the live Mode 1 run.** |

## P3 progress

| Item | Status | Notes |
|---|---|---|
| R9 chip timing beyond the horizon | **window fix done** — holds/patience/FH blank search bounded by the current chip window (GW19 for the first set). Still structure-based past the xP horizon (no chip-value projection); that remainder is documented, not built. | `fpl/optimize/chips.py` |
| R7 attack double-count | **done** — `xp.team_goal_scales()` caps each player's fixture goals at his share of the side's expected total (the opponent's `xgc`), the same rule as `simulate._allocate`; xP and simulation now agree. Assists are not capped (no team-assist total exists). | `fpl/model/xp.py` |
| Proper probability scoring | **partial — C6-4 fixed**. Brier scores and reliability bins are implemented for single-fixture player-gameweeks. Blanks naturally have no joined outcome and doubles are explicitly excluded because gameweek-total minutes cannot settle a per-fixture `p_60` forecast. CRPS still needs stored distributions. | `fpl/backtest/ledger.py`, `fpl/model/xp.py` |
| R6 cold starts / stale overrides | **done (three of four parts)** — exposure-weighted positional priors (`scoring.per90_rates`); form weight capped by minutes/90 (`scoring.form_weight`); stale overrides decay by half per freshness budget of extra age (`minutes.override_trust`). NOT done: a cross-league prior for newcomers (needs external data). | `fpl/model/scoring.py`, `fpl/model/minutes.py` |
| R4 confidence into the distribution | **open for epistemic uncertainty; C6-5 and decision consistency fixed**. `start_evidence` is emitted but intentionally does not alter one-event Bernoulli draws. The biased random denominator was removed, restoring the modeled `p_60` marginal. Gain diagnostics now use the final XI, bench, captain and vice with appearance-aware scenario scoring. Persistent posterior worlds for starts, rates and team strengths remain open and are R8-adjacent. | `fpl/model/minutes.py`, `fpl/model/simulate.py`, `fpl/pipeline.py` |
| R5 event-specific team-coherent minutes | **slice done** — `minutes.reconcile_team_starts` scales a side's starts down to eleven (never up). Open: per-event minute distributions, depth chart, injury redistribution to named deputies, override event ranges. | `fpl/model/minutes.py` |
| R8 multi-period MILP | **implemented as a challenger** — exact ownership, weekly XI/captain/bench, bank, original selling values, FT rollover and hits across the xP horizon. Live use is opt-in; walk-forward compares it by default. | `fpl/optimize/multiperiod.py`, `fpl/pipeline.py`, `fpl/backtest/replay.py` |
| R10 BPS rebuild | **done for the simulation layer, cross-validated against real fixtures; the `build_xp`/Mode 1 calibration attempt was reverted as statistically wrong — see design section below** | `fpl/model/bps.py`, `fpl/model/simulate.py`, `fpl/model/xp.py` |

### Behaviour changes in this session, for the next live run

- **Mode 1 now decides on expected points** (R1). `optimizer.rank_squad: true`
  restores rank-decided squads. The report states which objective decided and
  treats rank as diagnostic when expected points won.
- **The live reported XI is re-picked for the week** (B8). Replay, rank metrics
  and gain diagnostics all use that weekly XI. Rank scenarios also apply the
  reported bench order and captain/vice decisions under simulated appearances.
  `build_lineup(exact=False)` still exposes the horizon XI when explicitly needed.
- **Player goals are capped at the team total** (R7): strong attacks project a
  little lower than before when their players' xG summed past the side's.
- **A teammate's xP moves when you override one player's minutes** — intended
  (team-total cap). Other clubs do not move.
- **Stale overrides fade** rather than applying at full weight; the flag says
  the weight applied.
- **Chip holds stop at GW19 in the first half** (R9).
- The transfer section shows *"comes out ahead of holding N% of the time"*
  from the scenarios. Expected gain still decides. The diagnostic now uses the
  exact final lineups and appearance-aware armband/autosub rules; persistent
  epistemic start uncertainty remains open under R4.

### R8 — rolling multi-period transfer MILP: implemented (2026-09-18)

Why: the weekly solver picks one move now and holds a fixed 15 for the
horizon. Future free transfers, bank, selling values and future buys do not
exist in it, so a small positive gain spends an FT that had option value, and
a gain in GW+4 is credited now even if the move could wait.

Implemented shape (mirrors open-fpl-solver; 4–6 gameweeks, never 38):

- Variables per player `p` and gameweek `g`: `owned[p,g]`, `start[p,g]`,
  `tin[p,g]`, `tout[p,g]`, captain and ordered bench slots. Per gameweek:
  `bank[g] >= 0` plus a one-hot `(FT before, moves)` state.
- Constraints: `owned[p,g] = owned[p,g-1] + tin[p,g] - tout[p,g]`; squad
  composition 2/5/5/3 and club cap per `g`; `bank[g] = bank[g-1] + Σ sell·tout
  - Σ price·tin` with selling value from `transfers.selling_price` against
  recorded purchase prices for the initial squad and **purchase price =
  price at buy** for players bought inside the horizon (price changes are not
  modelled); exact FT transitions use an enumerated state table, so the MILP
  cannot exploit a loose min/max relaxation to invent FTs or avoid hits.
  An inherited club overage may be held, but any transfer restores the cap.
- Objective: `Σ_g decay^(g-g0) · (Σ start·xp_gw{g} + captain + bench terms
  - hit_cost·hits[g]) + terminal value` where terminal value =
  `ft_value · ft[G]` with `ft_value` a config knob to be estimated by replay,
  not guessed (start at 1.5 xP).
- Pool pruning keeps the top configurable 150 by `xp_horizon`, every owned
  player, and a positional safety set. Exact-score ties prefer fewer moves.
- The first move is exposed through `TransferPlan`; future moves are labelled
  contingent and re-solved next week. Its gain baseline is a fresh rolling
  solve forced to wait one week, not an unrealistic never-transfer path.
- `optimizer.multi_period_transfers: true` opts the live pipeline in. The
  default stays false until validation. `run_walkforward.py` now includes a
  separate executable `multiperiod` policy by default beside `expected`, so
  promotion can be based on sequential point-in-time regret rather than one
  gameweek or in-sample fit.
- Regression tests pin delayed moves, the FT cap, paid hits, selling-value
  affordability, weekly squad/XI legality, report semantics, replay execution,
  the opt-in live route, and (fixed 2026-09-19, two rounds) that a
  `differential`/`template` ownership tilt never reports a raw-xP-negative
  transfer as the recommendation — `optimize_multi_period()` falls back to a
  `wait` baseline solved with the tilt zeroed (`ownership_weight=0.0`)
  whenever the tilt-chosen path's raw gain against that raw-optimal baseline
  is negative. See the R8 Codex review section above for both findings and
  why the baseline itself had to be re-solved without the tilt, not just
  compared after the fact.

Deliberate limits: no chips inside the rolling MILP and no forecast price
changes. The terminal FT value (default 1.5 xP) is explicitly provisional and
must be estimated from uncontaminated replay before live promotion. While the
rolling policy is enabled, the old fixed-squad Wildcard-quality surplus is
suppressed rather than subtracting unlike objectives; the independent chip
timing heuristics still run. A cached production-sized smoke solve (659-player
GW6 frame, five events, pool pruned internally) completed in 5.9 seconds and
returned a legal no-hit path. This is a performance/legality check, not an
outcome validation. Full repository verification after R8: **765 passed, 1
existing warning**.

### R10 — projected BPS from simulated events: **done (2026-09-18)**

Why: bonus is carried as historical `bonus90`, fixture-scaled, and drawn
independently per player. Only the top three BPS in a match score bonus, so
bonus is a within-match ranking, not an independent rate.

Implemented in `fpl/model/bps.py` (`score_side_bps`, `award_match_bonus`) and
wired into `fpl/model/simulate.py`'s match loop. `score_side_bps` builds an
APPROXIMATE per-scenario BPS from the events the simulator already draws
(appearance, goals by position, assists, clean sheet, saves, cards, DC) --
not the official table's ~25 components; tackles, clearances/blocks/
interceptions, recoveries, crosses, key passes, dribbles, shots, passing
tiers, fouls and errors are not modelled and are not folded into the
constants (documented residual, same spirit as the "Open residuals" section
below). `award_match_bonus` ranks BOTH sides of the fixture together per
scenario and awards 3/2/1 with FPL's exact tie rule (a tied tier's SIZE
advances the next rank, not 1) via one vectorised "how many players are
strictly ahead of me" count -- verified against all three of FPL's own
documented tie examples.

Two bugs found and fixed while building this, both about who is even
ELIGIBLE for bonus: a non-appearing squad member (the simulator carries a
club's whole roster, not just who played) scored a genuine 0 on every BPS
component, which tied him with every other non-appearer for the match lead;
and marking those non-appearers `-inf` alone was not enough, since in a
match with fewer than three real scorers the tied `-inf` group could still
inherit whatever rank was left over -- fixed by explicitly zeroing bonus
wherever BPS is `-inf`, regardless of computed rank.

Consequence, verified and expected, not a bug: total bonus per match now
averages ~6 (the real ceiling, a little higher with ties), where the old
independent draw had none. A diagnostic with a uniform `bonus90` across 36
players showed the OLD analytic `bonus90`-rate summing to 16.8 across the
match -- nearly 3x the true total -- so the new simulated mean now diverges
from that analytic estimate; `tests/test_simulate.py`'s mean-agreement tests
exclude bonus from the comparison (via a new `bonus` field on
`simulate_event_detailed`) rather than have their tolerances loosened past
the point of still catching a real regression in the other components.

### R10 re-review (2026-09-18) — Codex found the fix hadn't reached the primary decision path, and three coefficients were stale; **finding 1's fix was itself wrong and reverted, see the section below**

The re-review's core point: R10 as first landed only touched
`model.simulate` (the rank layer's diagnostic simulation). `build_xp`'s
`xp_next1` -- Mode 1's DEFAULT squad-selection objective -- still called
`expected_bonus_for()` uncalibrated, and `moment_match()` rescales the
simulation back to `xp_next1`'s mean, so neither the primary decision nor
the rank layer (once moment-matched) actually reflected the fix.

| Finding | Fix |
|---|---|
| High — `build_xp`'s bonus term (hence `xp_next1`, hence Mode 1's default pick, hence `moment_match`'s target) was still the old, uncalibrated independent-rate estimate. | `expected_bonus_for`/`expected_bonus` take a `position` and apply `BONUS_CALIBRATION` (GKP 0.42, DEF 0.35, MID 0.41, FWD 0.80) -- how much of the independent-rate prediction survives real match-wide competition, derived by running the corrected `score_side_bps`/`award_match_bonus` ranking over a large simulated league built from domain-knowledge bonus90/xg90/xa90/dc90/saves90/cards90 tiers (NOT fitted to precise historical rates -- a full refit against the production `blended_rates` pipeline's real historical output remains a follow-up). `build_xp` now passes `position=pos` at its one call site. Omitting `position` keeps the OLD, uncorrected number rather than a silent change for any caller without one. |
| High — the scorer omitted the GKP/DEF goals-conceded BPS penalty entirely (`conceded_on` was already computed in `_score_side` for the separate FPL POINTS penalty) and had two stale season-specific values: `BPS_SAVE` was a 2025/26-era inside/outside-box blend (2026/27 changed saves to a flat 2 BPS + 1 for a "big chance save" the simulator has no signal for -- now 2.0, the guaranteed base) and the DC blend used a single flat weight (2026/27 also halved the CBI rate from 1-per-2 to 1-per-3). | Added `BPS_CONCEDED = -4.0` (per goal, GKP/DEF, not per two) using the already-simulated `conceded_on`. Corrected `BPS_SAVE` to 2.0. `BPS_DC_ACTION` is now per-position (`{"GKP": 0.5, "DEF": 0.7, "MID": 0.8, "FWD": 0.5}`), informed by the real-fixture validation below. Penalty-goal handling (always 12 BPS regardless of position) remains a documented residual: the simulator's `_allocate` has no penalty-vs-open-play distinction to key off. |
| Medium — no predictive validation existed, only ranking-mechanics tests. | New `scripts/validate_bps.py` reads cached `element-summary` history (real per-fixture minutes/goals/assists/cards/saves/DC/conceded, and FPL's own real `bps`/`bonus`) and reports exact-recipient match rate, Jaccard, recall and BPS MAE, overall and by position. Against real GW1-4 fixtures (40 matches): **42.5% exact bonus-recipient match** (up from 37.5% before the per-position DC weights), **0.675 mean Jaccard**, **0.76 recall**, **3.28 BPS MAE** (GKP 2.41 / DEF 3.39 / MID 3.49 / FWD 2.60). The residual is structural, not a further tuning target: FPL's public `element-summary` API does not expose crosses, key passes, dribbles, shots, passing-accuracy tiers, fouls or errors at ALL (confirmed by inspecting its full field list), so several official BPS components cannot be reconstructed from this data source regardless of coefficient choice. A 6-fixture frozen sample (`tests/data/real_bps_sample.json`, since `data/cache` is gitignored) backs a permanent regression floor in `tests/test_bps.py`. |
| Low — this file was internally contradictory: said bonus was still independent in one place (P3 table, R3 row) and "not started" in another (P3 table, R10 row) while the design section above said done. | Corrected both stale rows. |

Not done at the time of that section: a full `bonus90` recalibration
against the production `blended_rates` pipeline's real historical output.

### R10 third review (2026-09-18) — Codex checked finding 1's fix against real data and it was wrong; reverted. Findings 2 and 3 fixed properly

**Finding 1's fix (the `BONUS_CALIBRATION` table above) was a genuine
statistical error, confirmed and reverted, not merely refined.** `bonus90`
is derived (`model.scoring.RATE_SPECS`, `"bonus90": "bonus"`) from the REAL
`bonus` field FPL actually awarded each player -- i.e. it is ALREADY the
outcome of real match-wide BPS competition, a historical per-90 average of
points that survived it, not an independent-rate assumption needing a
further "how much survives the match" discount. Projecting it forward
linearly (`bonus90 * minutes-share * fixture-scale`) is an ordinary
extrapolation of an already-calibrated statistic. The calibration factors
had been derived from a synthetic diagnostic with an unrealistic UNIFORM
`bonus90` across every simulated player (not how real `bonus90` is
distributed across a real pool) and were never checked against real data
before landing. Checked now: applied to the real 2026/27 GW1-4 bootstrap,
those factors cut correctly-awarded bonus totalling ~6.4-6.5/match down to
~3.0/match -- 46% of the true total survived.

| Finding | Fix |
|---|---|
| High (Codex, 3rd pass) — `BONUS_CALIBRATION` double-corrected an already match-constrained statistic; see above. | Removed entirely: `BONUS_CALIBRATION`, `expected_bonus_for`'s/`expected_bonus`'s `position` parameter, and `build_xp`'s `position=pos` argument at its one call site are all gone. `xp_next1`'s bonus term is back to a plain linear projection of `bonus90`. A genuine correction, if one is ever warranted, needs fitting against the production `blended_rates` pipeline's real historical output with an out-of-sample check -- left open, undone. |
| High (Codex, 3rd pass) — the reported validation never exercised `BONUS_CALIBRATION`/`expected_bonus_for`/`build_xp` at all (it validated the simulation's event-to-BPS approximation only), and the per-position DC weights were selected AND evaluated on the same 4 gameweeks -- in-sample model selection presented as validation. | Added `scripts/validate_bps.py::logo_cv()`: proper leave-one-gameweek-out cross-validation over a small candidate grid. The SAME weight set (`{"GKP": 0.5, "DEF": 0.75, "MID": 0.85, "FWD": 0.55}`) was selected on every one of the 4 training folds -- not fold-dependent, which would have signalled overfitting -- with mean held-out MAE 3.24, close to in-sample (3.26) and consistently below the uniform-0.6 baseline (3.49) on the SAME held-out folds. Production `BPS_DC_ACTION` now uses this cross-validated set. |
| Medium (Codex, 3rd pass) — three validation-harness bugs: `python scripts/validate_bps.py` failed with `ModuleNotFoundError` (missing the `sys.path` bootstrap every other script here uses); the frozen regression test computed ONE clean-sheet flag for a whole match from the first player's side, so a losing side's defenders could be credited with a clean sheet whenever the first player in the group happened to be on the winning side; the yellow/red card split was conflated into one -3 weight even though real historical data has both counts separately. | Added the `sys.path` bootstrap. `tests/test_bps.py`'s frozen test now scores each side (grouped by `was_home`) separately, matching how `score_side_bps` is actually used in production. Both the script and the frozen test weight yellow (-3) and red (-9, new `BPS_RED_CARD`) separately. |

With finding 1 reverted, the validated real-fixture numbers (updated after
the cross-validated DC weights): **45.0% exact bonus-recipient match**,
**0.692 mean Jaccard**, **0.777 recall**, **3.24 BPS MAE** (GKP 2.41 / DEF
3.35 / MID 3.43 / FWD 2.57) -- all for `score_side_bps`/`award_match_bonus`,
the SIMULATION's diagnostic layer. `build_xp`'s `xp_next1` (Mode 1's
default decision) is intentionally uncalibrated bonus90, per the reversal
above.

Suite after this correction: **746 passed, 1 warning**.

### R10 fourth review (2026-09-18) — CV used the wrong dc feature; clean sheet used the wrong scope. Both fixed and verified

1. **High, FIXED.** `scripts/validate_bps.py` had computed its `dc` feature
   as `clearances_blocks_interceptions + recoveries + tackles`, but
   production `dc90` (`model.scoring.RATE_SPECS`) is sourced from the
   `defensive_contribution` field directly -- a DIFFERENT, position-
   dependent aggregate. Confirmed against the cached GW1-4 history: GKP's
   `defensive_contribution` is always 0 (815 in the validator's old sum, 0
   in production -- goalkeepers are not DC-threshold-eligible,
   `DC_THRESHOLD` in `xp.py` sets their bar at 99, effectively
   unreachable); for DEF, `defensive_contribution` = CBI + tackles ONLY,
   excluding recoveries (2,584 vs the old validator's 3,771); MID/FWD
   happen to match exactly, since their official DC definition does
   include recoveries. Fixed: the validator now reads
   `defensive_contribution` directly; `logo_cv` re-run against the
   correct feature selected the SAME candidate on every fold again
   (`{"GKP": 0.4, "DEF": 0.8, "MID": 0.9, "FWD": 0.6}`, mean held-out MAE
   3.61 vs uniform-0.6's 3.98), and production `BPS_DC_ACTION` now ships
   these corrected weights.

   This ALSO surfaced a genuine, previously undiagnosed finding: GKP's
   BPS approximation undershoots the real value by ~6-9 points per
   appearance, fairly flat regardless of save count -- too large and too
   save-count-independent to be the missing save-BPS component alone.
   Likely dominant cause: the passing-accuracy tiers (30+ attempts,
   2-6 BPS), which goalkeepers routinely clear via goal-kick distribution
   and which this codebase has no pass-attempt data to model at all.
   Documented in `BPS_SAVE`'s docstring rather than guessed at with an
   arbitrary correction constant.
2. **High, FIXED, FPL rule confirmed via the official rules page.** Clean
   sheet credit used `conceded_team == 0` (the match's FULL final score)
   even though `_score_side` already computes `conceded_on` (goals
   conceded specifically WHILE THIS PLAYER was on the pitch) for the
   separate goals-conceded penalty. Confirmed: the official rule is no
   goal conceded WHILE ON THE PITCH, not the team's final result -- a
   defender subbed at 60' keeps his clean sheet even if his side concedes
   afterwards. Fixed in both the FPL points term (`_score_side`) and
   `score_side_bps` (whose `clean_sheet` parameter changed shape from
   per-side `(n_sims,)` to per-player `(n_players, n_sims)` accordingly):
   clean sheet is now `conceded_on == 0`. `validate_bps.py` and the
   frozen test now read FPL's own `clean_sheets` field (which already
   implements the correct rule) instead of re-deriving it from the final
   score.
3. **Medium, documented more explicitly, not resolved.** `BPS_SAVE`'s
   exact 2026/27 rule has now been re-checked twice across review rounds
   with different results each time; its docstring now says so plainly
   (genuinely uncertain, not asserted) rather than presenting either
   fetch as settled. No shot-location or big-chance-faced data exists in
   this codebase to implement it exactly regardless of which reading is
   correct.
4. **Medium, FIXED as a documentation-honesty correction.** The `logo_cv`
   candidate grid was itself designed after looking at all 4 cached
   gameweeks, so holding out one gameweek per fold does not make the
   grid's hypothesis space independent of the data -- it shows stability
   WITHIN that grid, not absence of overfitting to it. `logo_cv`'s
   docstring and the `BPS_DC_ACTION` comment now call this exploratory/
   grouped CV explicitly, not full out-of-sample validation; genuinely
   fresh gameweeks, once available, are the real prospective test.

Regenerated `tests/data/real_bps_sample.json` with the
`defensive_contribution` and `clean_sheets` fields the fixes need. New
regression tests (`score_side_bps`'s per-player clean sheet directly; a
full `simulate_event` statistical check that a partial-minutes defender's
clean-sheet rate exceeds the team's own full-match clean rate) verified to
fail against pre-fix code via `git stash`. Live validation after all four
fixes (`scripts/validate_bps.py`): 45.0% exact bonus-recipient match, 0.690
mean Jaccard, 0.769 recall, 3.66 BPS MAE overall (GKP 6.26 / DEF 3.83 / MID
3.43 / FWD 2.57 -- GKP's number is the newly-diagnosed, real, currently
unfixable residual above, not a regression in the fix).

Suite after this review: **748 passed, 1 warning**.

### R10 fifth review (2026-09-18) — the clean-sheet fix broke teammate correlation; analytic xP still diverged; GKP save value was under-fit. All fixed and verified

1. **High, FIXED.** The 4th review's `conceded_on` fix independently
   thinned EACH player via `rng.binomial` on an `(n, n_sims)` array --
   which still draws one INDEPENDENT sample per player-scenario cell, so
   two defenders with the IDENTICAL playing window could disagree about
   the SAME goal. Confirmed: two identical 60-minute players disagreed in
   44.4% of scenarios (exactly `2*p*(1-p)` at `p=60/90`), undermining the
   whole simulator's purpose (teammate correlation). Fixed with a new
   `_conceded_on()`: the match's conceded goals now get ONE shared
   simulated timing per scenario (each goal a Uniform(0,1) match-fraction,
   common to every player on that side), and each player counts how many
   of those SAME timed goals fall in his own interval (`[0, share]` if
   started, `[1-share, 1]` if he came on as a sub). Re-verified: 0%
   disagreement between identical players, same marginal mean as before.
2. **High, FIXED.** `xp_for_fixture` still used `p_cs * CS_PTS * p_60`
   (`p_cs` = the FULL match's clean-sheet probability), inconsistent with
   the simulation's now-correct per-player rule. Measured gap for a
   certain 60-minute defender facing xgc=2: simulation 2.619 vs analytic
   2.107 (0.512pts). Fixed with `p_clean_sheet_over_minutes()`, modelling
   conceded goals as a homogeneous Poisson process so a `minutes`-long
   window's clean-sheet probability is `exp(-xgc*minutes/90)` --
   deliberately NOT routed through the existing `minutes_branches` helper,
   since that treats "started" as always reaching the full `m_start` with
   no separate early-withdrawal chance, which this term's hard 60-minute
   cutoff is sensitive to in a way the OTHER (smoothly-integrated)
   threshold terms are not; the new function instead weights by
   `min(p_60, p_start)` directly, mirroring the simulation's own
   `p60_given_start`. Re-measured: gap down to 0.0016pts (Monte Carlo
   noise floor).
3. **Medium, FIXED.** The prior regression test inferred a "clean-sheet
   rate" from `samples - 2`, which still carried bonus and other
   components and did not isolate the event (reconstructing the pre-fix
   formula also happened to pass it). `conceded_on` is now exposed
   directly on `simulate_event_detailed`'s output, so the event
   (`conceded_on == 0`) is asserted directly; a SEPARATE new test covers
   finding 1 specifically (two identical teammates' `conceded_on` must
   agree in >99.5% of scenarios).
4. **Medium, FIXED with a data-fitted value.** The GKP residual diagnosis
   was wrong on two counts: it is NOT "fairly flat regardless of save
   count" (real correlation 0.477, ~0.894 extra BPS per save, 95% CI
   [0.52, 1.26], residual rising from ~3.86 at zero saves to ~8.50 at
   five), and the inside-box save bonus is not merely "possibly" part of
   the rules. `BPS_SAVE` raised from 2.0 to 2.9 (base + the fitted slope,
   rounded); GKP MAE improved 6.26 -> 3.82 on the live validation. The
   save-count-INDEPENDENT part of the residual (~3.6-3.86 at zero saves)
   remains a separate, likely passing-accuracy-driven gap with no data to
   close it.
5. **Low, FIXED.** `validate_bps.py` still called `logo_cv` an "honest
   OUT-OF-SAMPLE check" in two places, contradicting the exploratory-CV
   caveat the 4th review added. Both reworded to state plainly that the
   candidate grid was designed after seeing all 4 cached gameweeks.

Live validation after all five fixes: 42.5% exact bonus-recipient match,
0.684 mean Jaccard, 0.769 recall, 3.45 BPS MAE overall (GKP 3.82 / DEF 3.74
/ MID 3.40 / FWD 2.58). DC weights unchanged by the save-value fix (`{"GKP":
0.4, "DEF": 0.8, "MID": 0.9, "FWD": 0.6}`, still selected on every
leave-one-gameweek-out fold, mean held-out MAE 3.45 vs uniform-0.6's 3.82).

Suite after this review: **749 passed, 1 warning**.

### R10 sixth review (2026-09-18) — scoreline tail and save-calibration leakage fixed

1. **Medium, FIXED.** `_conceded_on()` allocated only eight shared goal-time
   slots. A simulated ten-goal scoreline therefore became eight goals for a
   full-match player's conceded-points and BPS calculations. The timing axis
   is now sized from the realised maximum in `conceded_team`, so every drawn
   goal is timed without changing ordinary-case cost. A direct regression
   test pins ten conceded goals to ten for a 90-minute player.
2. **Medium, FIXED.** `BPS_SAVE=2.9` had been estimated from all GW1-4 rows,
   then reused while each of those weeks was described as held out. The
   validation script now contains the actual reproducible estimator:
   goalkeeper `real BPS - approximate BPS at the official 2.0 base` is
   regressed on saves with an intercept, and an HC3-robust 95% interval is
   reported. Full-sample exploratory fit: n=80, residual/save correlation
   0.477, extra BPS/save 0.894 (HC3 95% CI 0.500-1.287), total fitted save
   value 2.894 versus the shipped rounded 2.9. Inside `logo_cv`, that save
   value is now re-fitted from each TRAINING partition before DC selection
   or holdout scoring (fold values 2.91 / 2.72 / 2.89 / 3.07). Mean held-out
   MAE remains 3.45 versus 3.82 for uniform DC weights. This is still
   exploratory because the model family was designed after inspecting the
   same four weeks; fresh gameweeks remain the prospective test.
3. **Low, FIXED.** Identical playing windows now assert exact equality with
   `np.array_equal`; the test no longer permits a nominal 0.5% rate of an
   event that should be impossible.
4. **Low, FIXED.** `model.bps` now states the current official save rule
   directly (2 for any save, +1 inside the box, +1 for a big-chance save)
   rather than retaining the previous contradictory “uncertain” wording.

Live full-sample validation is unchanged: 42.5% exact bonus-recipient match,
0.684 mean Jaccard, 0.769 recall and 3.45 BPS MAE. Suite after this review:
**752 passed, 1 warning**.

### Open residuals worth knowing

- Assists are still drawn independently of goals in the simulation.
- The oracle remains a deliberately impossible one-week rebuild. The new
  `multiperiod` row is the fair executable rolling-policy comparison; it is a
  challenger until enough uncontaminated gameweeks exist.
- No cross-league prior for newcomers (R6 part four); no cohort picks for the
  rival field (R2 second half); no per-event minutes (R5 remainder).

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

P2 is complete (B6, B7, B8, R1, R2-achievable, R3), including
appearance-aware vice-captain inheritance and legal autosubs in candidate rank
scoring. Of P3, R5 (team-start slice), R6 (three parts), R7, R9 (window fix),
single-fixture Brier scoring, and R10 (bonus from ranked simulated BPS) are
done -- see the P3 progress table. No original audit item remains wholly
unimplemented. Remaining in part: R2 (cohort picks), R4 (persistent start/rate/team-strength
posteriors), R5 (event-specific minutes), R6 (newcomer prior), R10 (analytic
`bonus90` recalibration, `--legacy-bonus` comparison flag), and CRPS/
distribution storage.

## Recommended implementation order (steps 1–5 done 2026-09-18)

1. ~~Add failing regression tests for RB4-RB7, then fix those live-decision paths.~~
2. ~~Fix confirmation legality and exact forecast identity (RB8-RB9).~~
3. ~~Wire snapshots into replay and restore the production horizon/captain decision
   (RB1-RB3), using versioned action-time snapshots (RB10).~~
4. ~~Make Free Hit restoration work offline from `base_*` state (RB11).~~
5. ~~Run the full suite and one end-to-end historical replay whose snapshot values
   deliberately differ from current cache.~~
6. ~~Close RR1–RR7.~~ Done at `880fbc9..dfd6f10`; third-review findings done after.
7. ~~Close P2 — B8 → B7 → R3 — and make the rank scorer consistent with the
   final XI, bench and armband.~~
8. ~~Implement R10 -- bonus awarded by ranking simulated BPS within a match,
   with FPL's tie rule, instead of an independent per-player draw.~~
9. ~~Implement R8 as a rolling-policy challenger with exact FT/bank/hit state.~~
10. **Next:** accumulate uncontaminated replay weeks and estimate R8's terminal
    FT value before promoting it; then add persistent multi-week epistemic
    worlds. Do not tune from the current contaminated sequential replay numbers.

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

Codex's earlier passes were review-only. Review 7 changed production rank,
simulation, pipeline and report code, added regressions, and reconciled this
handoff.

## Performance refactor: fetch and cache (2026-09-22, in progress)

Separate initiative, isolated in worktree `.claude/worktrees/perf-fetch-and-cache`
(branch `worktree-perf-fetch-and-cache`, off `master` @ `c3a5c8c`). Not yet
merged; this section tracks it in case the session is interrupted.

**Why:** GW6 profiling found a weekly run is slow two ways — a post-gameweek
refresh crawls ~667 element-summaries one at a time behind a 1 s throttle
(11+ min), and every run's cache lookups `glob()` the whole cache directory
per call (22s/40.6s of a cache-only GW6 run). Spec:
`docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md` (2 review
rounds, both addressed — see its Problem/Goals sections). Plan:
`docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md`.

**Approach:** cache slug→paths index (mtime-validated, self-healing) +
concurrent element-summary fetch (bounded thread pool, shared token-bucket
limiter, 429/5xx retry with Retry-After, per-player failure containment,
cancellable) + optional xP vectorization.

**STATUS AS OF 2026-09-22 20:50 — HANDING OFF TO CODEX FOR TASKS 5-7.**
Everything below this line is written for a fresh agent (Codex) picking this
up cold. Tasks 1-4 were executed by Claude via
superpowers:subagent-driven-development (SDD) and are DONE, committed,
reviewed clean. The user asked to stop here and hand the rest to Codex —
this is not an interruption, it's a deliberate handoff point.

### Where everything lives

- **Spec** (binding authority, read this if the plan and code ever seem to
  disagree): `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`
- **Plan** (the full 7-task implementation plan, complete code for every
  task): `docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md`
- **SDD ledger** (full task-by-task history, every review verdict, every
  ruling made so far, with reasoning):
  `.superpowers/sdd/2026-09-22-fetch-and-cache-perf-plans/progress.md`
  **NOTE the `-plans` suffix** — an earlier attempt manually created a
  workspace dir WITHOUT that suffix
  (`.superpowers/sdd/2026-09-22-fetch-and-cache-perf/`, containing only a
  stray `plan-path` marker, no ledger) before discovering that this
  environment's `sdd-workspace` helper script resolves `git rev-parse
  --show-toplevel` differently inside a subshell than in a direct shell call
  (Windows-drive-letter path vs. POSIX path), which broke its own
  plan-ownership matching. **If that other directory still exists, ignore
  it — it is stale and empty of real content.** The ledger's own "Tooling
  note" and "Ruling: commit attribution line" entries have the full story.
- **Pre-extracted task briefs**, each containing the EXACT code to write —
  Codex should read these directly rather than re-deriving from the plan:
  - `.superpowers/sdd/2026-09-22-fetch-and-cache-perf-plans/task-5-brief.md`
    (Phase 2a: `TokenBucket` rate limiter — NOT YET STARTED)
  - `.superpowers/sdd/2026-09-22-fetch-and-cache-perf-plans/task-6-brief.md`
    (Phase 2b: concurrent `element_summaries`, the big one — NOT YET STARTED)
  - Task 7 (optional xP vectorization) has no pre-extracted brief; it's in
    the plan document itself under "### Task 7" — **do not start it without
    asking the user first**, see below.
  - Tasks 1-4's briefs/reports/review-packages are also in that directory,
    kept as historical record — task-1 through task-4 briefs, reports,
    review-packages, plus `task-3-fix-review-package.md` for the one fix
    round Task 3 needed.

### Current repo state (verify with `git log --oneline -5` before trusting this)

- Worktree: `.claude/worktrees/perf-fetch-and-cache`, branch
  `worktree-perf-fetch-and-cache`, branched from `master` @ `c3a5c8c`.
- HEAD as of handoff: `edd452d` ("perf: index cache snapshots by slug instead
  of globbing per lookup").
- Working tree is clean (no uncommitted changes).
- `python -m pytest -q -p no:cacheprovider` → **793 passed, 3 xfailed**.
- `python scripts/bench.py golden check` → **golden: OK**.
- `docs/perf/README.md` has real numbers filled in for `baseline` and
  `phase1-cache-index` columns; `phase2-concurrent-fetch` and `phase3-xp` are
  still `—`, to be filled in by Tasks 6 and 7 respectively.
- Success criteria progress: **criterion 2** (cache-only GW6 run ≤ 20s) is
  MET — 34.26s → 14.52s. **Criterion 1** (refresh ≤ 3 min) is NOT YET MET —
  that's what Task 6 delivers.

### What's done (Tasks 1-4)

1. **Task 1** (`8fa9182`): `scripts/bench.py` benchmark harness, a
   clock-frozen GW6 "golden" decision + xP frame captured from unmodified
   master (`docs/perf/golden-gw6/`), baseline numbers.
2. **Task 2** (`18af003`): `tests/test_perf_contracts.py` — 4
   `xfail(strict=True)` tests pinning HOW the cache/fetch should scale
   (directory scans, in-flight requests, rate cap, real concurrency). One
   xfail was removed by Task 4; the other three are removed by Task 6.
3. **Task 3** (`4629607`, fixed at `edaf677`): parity + per-player
   containment tests for today's SEQUENTIAL `element_summaries`, so Task 6's
   rewrite has a safety net. **One fix round was needed**: the review found
   that `test_a_corrupt_fallback_snapshot_fails_only_that_player`, exactly as
   given in the plan's own Task 3 brief, didn't actually reach the code
   branch its name claimed (a corrupt file always raises during
   `Cache.get_fresh()`'s FIRST statement, before the network/fallback path is
   ever reached) — it silently duplicated another test. **Lesson for
   Codex:** the plan's given test code is not infallible; when a test's
   name/docstring claims to pin a specific branch, trace it against the real
   current code before trusting it, the way the Task 3 reviewer did. Full
   before/after and the traced fix are in the ledger and in
   `task-3-report.md`/`task-3-fix-review-package.md`.
4. **Task 4** (`edd452d`): replaced `Cache`'s per-lookup `glob()` with a
   slug→paths index, validated against the directory's mtime, self-healing
   on a vanished snapshot. Reviewed clean. **One forward-looking finding was
   raised and adjudicated (see below) — it's a Task 6 concern, not a Task 4
   defect.**

### What's left

- **Task 5** (not started): `fpl/data/throttle.py` — `TokenBucket`, a
  thread-safe, cancellable rate limiter with NO slot reservation (so a pause
  triggered by one 429 holds back threads that were already asleep waiting
  their turn — this was itself a fix from an earlier plan review, see the
  spec's revision history). Small, isolated, no dependency on anything else
  not already done. Brief: `task-5-brief.md`.
- **Task 6** (not started): replaces `fpl/data/client.py` wholesale with a
  concurrent `element_summaries` — bounded thread pool, the Task 5 limiter,
  429/5xx retry with `Retry-After` (parses both delay-seconds and HTTP-dates,
  never shortens it, gives up asking after 120s), per-player failure
  containment matching today's sequential semantics (this is what Task 3's
  tests exist to verify), and clean cancellation (Ctrl-C / a raising
  `progress` callback must stop workers within about one in-flight request's
  time, not hang). Also touches `fpl/config.py`, `config.yaml`, and 4
  production call sites. Ends with **Step 10, a manual live smoke test that
  needs real network access to FPL's API** — this can't be fully automated,
  budget for it. THIS IS THE HIGH-RISK TASK — concurrency, cancellation, and
  retry logic are all easy to get subtly wrong. Brief: `task-6-brief.md`.
  **Read the "How stopping works" section at the top of the brief before
  writing any code** — it's the part most likely to be gotten wrong.
- **Task 7** (optional, NOT YET APPROVED): vectorizes the xP threshold tail
  in `fpl/model/xp.py` for a further ~3s of the cache-only run. **The plan's
  own text says "Stop here... Task 7 runs only if they say go" after Task 6.
  Do not start Task 7 without explicitly asking the user first**, even if
  Tasks 5-6 go smoothly. This is not optional politeness — it's what the
  user/plan actually specified.

### Adjudicated findings Codex must carry forward

**From Task 4's review (Important, forward-looking — this is a Task 6 review
focus item, not a Task 4 defect):** Task 4 made `Cache` correct only under a
"one writer at a time" assumption (unsynchronized `self._index`/
`self._index_mtime`). The plan's Global Constraints already require that
Task 6 never violates this ("All cache and client-state mutation happens on
the main thread. Worker threads only perform HTTP") — Task 6's own brief
code has `_fetch_json` (the only method worker threads run) touch nothing
but the HTTP session, with every `cache.put`/`prune`/`newest` call happening
inside `_fetch_misses`'s main-thread loop after `wait(...)` returns. **When
Task 6 is implemented and reviewed, the reviewer must independently verify
by reading the actual code — not just trusting the brief — that no worker
thread ever touches `self.cache` directly.** If a future implementation
deviates from the given Task 6 code in a way that lets a worker thread call
into `Cache`, this becomes a real, blocking race condition, not a
theoretical one.

**Minor (deferred to final review, not urgent):** `Cache.prune()`'s guard
`if removed and self._index is not None:` has a dead half (`self._index` is
always a dict by that point) — cosmetic, harmless, low priority.

### Process notes for Codex

- This was executed with fresh-subagent-per-task + review-after-each-task +
  a bounded fix loop (max 5 rounds) for any Critical/Important finding,
  following the `superpowers:subagent-driven-development` skill. Codex is
  not required to replicate that exact process, but the QUALITY BAR each
  task was held to should carry forward:
  - Every task's changes are verified against the GW6 golden check
    (`python scripts/bench.py golden check` → must stay `golden: OK`) AND
    the full test suite (`python -m pytest -q -p no:cacheprovider`) before
    being considered done.
  - Commit messages in this branch so far end with
    `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (the plan's own
    example commit text says `Claude Opus 5.5` — that was written under an
    earlier model and was deliberately overridden throughout this session;
    Codex should use whatever attribution convention it's configured with,
    not literally copy either of those).
  - No new git worktrees, no branch switches — everything happens in-place
    on `worktree-perf-fetch-and-cache`.
  - No new third-party dependencies at any point in this plan.
- Update this handoff section (or add a new dated one below it, don't just
  silently overwrite) after each task, in case of another interruption.

### Codex continuation, 2026-09-22

Task 5 is complete at `f144b14`. The rate limiter was implemented after its
eight tests failed on the missing module. All eight pass; the full suite is
801 passed, 3 xfailed, and the GW6 golden check passes. Task 6 is next.

Task 6 is implemented at `0b77181`. The client now fetches element summaries
with four HTTP workers under a shared 5 starts/s cap; all cache reads, writes,
and client-state updates remain on the calling thread. Retries, Retry-After,
long server pauses, per-player cache failures, duplicate IDs, and cancellation
are covered by tests. Config settings are wired through all four production
client constructors. The final 820-test suite passes (one existing statistical
warning), and the frozen GW6 golden check is OK. The phase-2 benchmark reports
134.5s extrapolated refresh versus 648.4s baseline, and a 14.79s cache-only
GW6 run versus 34.26s baseline. The refreshed benchmark measures 4.96
requests/s. A live scratch-copy Mode-2 run with 150
snapshots removed completed in 27s and rendered a report; a second live run
interrupted during fetch returned promptly with one KeyboardInterrupt
traceback. No 429 errors were observed. The scratch copy remains at
`C:\Users\user\AppData\Local\Temp\fpl-perf-smoke-codex` because this
environment rejected its recursive deletion; it contains only copies and
live-smoke outputs, not changes to this worktree's `data/`.

The final review found two benchmark-integrity issues, now fixed: the refresh
result is the median of three runs, and both the golden driver and cache-only
GW6 benchmark fail immediately on any network attempt. The review also noted
that overlapping writers can race the cache index, but this is outside its
documented one-writer-per-directory contract; external-writer detection is
best-effort. No cache implementation change was made for that finding.

Implementation ruling: retry 429 and all HTTP 5xx statuses, matching the
spec's general “5xx” wording (the plan's sample code listed only selected
5xx statuses). The one-writer cache constraint was independently confirmed
against the implemented `_fetch_json` worker and `_fetch_misses` main-thread
integration. Final whole-branch review findings are resolved. Task 7 remains optional
and requires explicit user approval.

### Task 7 continuation, 2026-09-22

The user explicitly approved Task 7 after the phase-2 handoff. Implemented
the planned xP hot-loop change in `fpl/model/xp.py`: one vectorized Poisson
tail call, one fixture grouping by team, and no per-player fixture-row copy.
One parity test passed on the old code; three performance/interface tests
failed on the old code and pass after the change. The full suite passes
(824 tests, one pre-existing statistical warning), and the offline, frozen
GW6 golden check is OK. The three-sample benchmark measured the cache-only
GW6 run at 12.00s, down from 14.79s after phase 2 and 34.26s at baseline.
The simulated refresh result is 134.8s extrapolated versus 134.5s in phase
2, effectively unchanged as expected. This work remains on the feature
branch, not merged; the user's approval to implement now superseded the
spec's earlier post-merge timing. Fresh final review approved Task 7 with no
Critical or Important findings. One minor provenance caveat remains:
`phase3-xp.json` records the pre-commit HEAD (`395099b`) because the bench ran
before the implementation commit (`bb20389`); the measured code is that
implementation commit. Integration into `master` remains the user's choice.
