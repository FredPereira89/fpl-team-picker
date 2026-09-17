# FPL model and optimizer audit — 2026-09-17

## Scope and verdict

This review covers the production path in `fpl/model/`, `fpl/optimize/`,
`fpl/backtest/`, `fpl/data/`, `fpl/pipeline.py`, `fpl/state.py`, `fpl/cli.py`, and
the actual entry point `run_gameweek.py`. It was checked against the original
design, both earlier review documents, `docs/STATUS.md`, the current 2026/27
rules, and comparable FPL/DFS/forecasting work.

The core one-week legal squad MILP is good. Budget, 2/5/5/3 squad composition,
eleven starters, legal formation, three-per-club in the normal case, current
selling value, and the basic `-4 * max(0, transfers - free_transfers)` calculation
are encoded correctly. Blanks are zero, double-gameweek fixtures are summed,
and captaincy is now valued per projected gameweek.

The system is not yet safe to describe as finding the best legal FPL team across
a season. The largest risks are not small forecast errors:

1. the 2026/27 two-sets-of-chips rules are not represented;
2. Wildcard/Free Hit state and free-transfer accounting are wrong;
3. a recommended Wildcard or Free Hit does not produce the squad that chip should
   actually field;
4. the default rank-aware transfer path commonly never considers a two-transfer
   plan, and then judges the remaining five-week candidates only on this week;
5. the rank simulator does not simulate a coherent football match and is not on
   the same calibrated scale as the optimizer;
6. the validation harness is contaminated and does not replay a season's manager
   state, so it cannot validate transfers, hits, rank utility, or chips.

Until those are fixed, use expected points as the primary decision objective,
keep rank output diagnostic, and do not automatically act on chip advice.

### Severity scale

| Severity | Practical meaning |
|---|---|
| Critical | Can recommend an illegal/unavailable chip, corrupt persistent squad state, or invalidate the evidence used to trust the system. Plausible cost is a chip or many gameweeks: roughly 10–30+ points in a bad case. |
| High | Can choose the wrong transfer, hit, captain, or squad structure. Plausible cost is about 2–10 points in an affected decision and material rank over repeated weeks. |
| Medium | Usually changes marginal choices or becomes large only in a specific state. Plausible cost is about 0.5–4 points when triggered. |
| Low | Rare rule edge or presentation/reproducibility weakness; normally below a point, but still worth a regression test. |

Point ranges are decision-impact estimates, not promises; football outcome
variance is much larger than model-value differences.

### Disposition of previously documented findings

This section prevents the implementation team from reopening fixes that are
already present.

- **Verified fixed:** current-season form reaches `blended_rates`; selling-price
  math uses purchase price and half-profit; the 2/5/5/3, budget, formation and
  ordinary club-cap constraints are present; captain value is calculated by
  gameweek; bench slots have distinct weights; stepped saves/goals-conceded and
  defensive-contribution thresholds are integrated over minutes branches;
  planning runs no longer write state without `--confirm`; API chip names are
  canonicalized; prediction versions are retained; and fixture counts now extend
  beyond the xP horizon.
- **Known and still open:** new-signing cold start, no injury-minute redistribution,
  stale manual overrides still being applied, goalkeeper bias/projected BPS,
  fixed lineups across the horizon, and lack of a multi-period transfer model.
- **Partially fixed:** “chip timing sees past the projection horizon” is true only
  for fixture counts. The code does not project the value of those future weeks,
  does not stop the first chip window at GW19, and cannot see a future blank or
  double until FPL has assigned the postponed fixture an event.

The test suite passed (`python -m pytest -q`: **510 passed**), which is useful
regression evidence but also shows that several tests currently codify the wrong
chip/FT behavior.

## 1. Bugs

### B1 — chip inventory is one-per-season in code, but one-per-half in the game

- **Location:** `fpl/state.py:108-124`; `fpl/optimize/chips.py:156-162,
  220-260`; `fpl/pipeline.py:379-385`; `tests/test_state.py:230-236`.
- **What is wrong:** `chip_events` can record two Wildcards, but `chips_used` is a
  unique list of names. `advise_chips` receives only that name list and turns it
  into a set. Once any Wildcard, Free Hit, Bench Boost, or Triple Captain has
  been played, the second-half copy is suppressed for the rest of the season.
  Conversely, the advisor has no concept of first-half expiry. The official
  2026/27 rule is two of every chip, one in GW1–19 and one in GW20–38, with the
  first set expiring rather than rolling over. Free Hit and Wildcard are also
  unavailable in an entry's opening gameweek, and Free Hits cannot be played in
  consecutive gameweeks. See the [official 2026/27 chip rules](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627)
  and [FPL FAQ](https://www.premierleague.com/en/news/4661030).
- **Concrete fix:** make `chip_events` the source of truth. Add
  `chip_window(event) -> 1|2` and `chip_available(name, event, chip_events)`.
  Validate one use per `(chip, half)`, GW1 restrictions, the GW19/GW20 Free Hit
  restriction, and one active chip per event. Remove `chips_used` from decision
  logic or derive it only for display. Pass dated events through `pipeline.run`.
- **Regression tests:** first Wildcard at GW5 still permits one at GW25; an
  unused first-half chip is unavailable at GW20; FH19 blocks FH20 but not FH21;
  WC/FH cannot be advised in the opening gameweek.
- **Severity:** **Critical** — it can forfeit or hide four second-half chips,
  easily 20+ points of season value.

### B2 — Wildcard and Free Hit incorrectly create an extra free transfer

- **Location:** `fpl/state.py:153-167,187-209`; tests that currently assert the
  wrong rule at `tests/test_state.py:64-69,273-276`.
- **What is wrong:** for a Wildcard or Free Hit, `used` becomes zero and the code
  returns `current_balance + 1`. FPL says the free transfer received for the chip
  gameweek is consumed by activation; previously banked transfers are retained.
  The following gameweek's balance is therefore the same total balance, not one
  higher. The official FAQ gives the explicit example that two saved transfers
  remain two after a Wildcard.
- **Concrete fix:** for `wildcard`/`freehit`, return a next-gameweek balance equal
  to `state.free_transfers` (capped at five), with a separately named value for
  saved FTs if the UI needs it. Correct `reconcile` and rewrite the two tests.
- **Severity:** **High** — it silently authorizes one transfer that actually
  costs four points in the gameweek after every WC/FH; repeated across two sets,
  the direct exposure is up to 16 points.

### B3 — confirming a Free Hit destroys the permanent squad and purchase-price state

- **Location:** `fpl/cli.py:32-60,118-147`; `run_gameweek.py:127-147`;
  state schema at `fpl/state.py:43-61`.
- **What is wrong:** `--confirm` writes the applied 15, bank and purchase prices
  as the permanent squad for every chip. For a Free Hit those are temporary. At
  the next deadline FPL restores the pre-chip squad and bank, but
  `resolve_current_squad` reads the previous gameweek's picks — the temporary FH
  squad — and the local state has already overwritten the only copy of the base
  squad and its purchase prices. Official rules also state that any bank change
  during the Free Hit is lost.
- **Concrete fix:** add explicit permanent/base state and optional
  `freehit_event`/`freehit_squad` display state. On FH confirmation, preserve
  permanent `squad`, `bank`, and `purchase_prices`; on the next planning run,
  ignore previous-GW picks if that event used FH and restore the saved base.
  Reconcile market prices without changing historic purchase prices.
- **Regression tests:** confirm FH with nine different players, then resolve the
  next GW and assert the original 15, bank, and purchase prices are restored.
- **Severity:** **Critical** — one FH can make every later budget and transfer
  recommendation start from the wrong team until manually repaired.

### B4 — chip advice is bolted on after optimization; it does not build the chip squad

- **Location:** `fpl/pipeline.py:136-170,343-385`;
  `fpl/optimize/chips.py:193-265`; `run_gameweek.py:130-147`.
- **What is wrong:** transfers/squad and lineup are finalized before
  `advise_chips`. A Free Hit recommendation still returns the ordinary permanent
  transfer plan instead of the best one-week £100m FH squad. A Wildcard computes
  a rebuild only to get a scalar `SquadQuality`, discards that rebuild, and
  returns the limited-transfer squad whose objective may include hit costs. If
  the user confirms the recommendation, the CLI defaults to that non-chip squad
  and to the advised chip. This is not merely a heuristic timing error: the
  output is not the action the named chip means.
- **Concrete fix:** model each available chip as a complete candidate action:
  hold/ordinary transfer path, WC permanent rebuild, FH temporary one-week
  rebuild plus base restoration, BB lineup value, and TC captain/vice value.
  Compare marginal value against the same no-chip baseline, return the selected
  action's actual squad/lineup/transfers/bank, and store permanent versus
  temporary state correctly. Until implemented, label chip advice “timing flag
  only” and require an explicit `--applied-chip` rather than defaulting to it.
- **Severity:** **Critical** — WC/FH are typically the largest decisions of the
  season; a non-optimized or state-corrupting chip can cost 10–30+ points.

### B5 — the default rank-aware transfer enumeration usually never reaches two transfers

- **Location:** `fpl/optimize/transfers.py:188-205`;
  `fpl/pipeline.py:190-202`; coverage gap in `tests/test_transfers.py:184-213`.
- **What is wrong:** the single global `k` quota is filled in ascending transfer
  count. The hold plan consumes one slot; a large player pool then supplies the
  remaining seven one-transfer alternatives, so the outer loop exits before
  `n=2`. With one FT, paid-hit plans are normally never shown to the default
  rank layer; with two or more banked FTs, coordinated free moves are normally
  missed too. The tests assert diversity and hit arithmetic but not that every
  allowed transfer count is represented.
- **Concrete fix:** first generate the best feasible plan for every `n` in
  `0..FT+max_paid_hits`; reserve one candidate per transfer count; then spend the
  remaining candidate budget on no-good-cut alternatives and rank globally.
  Alternatively use a stratified `k_per_n` and prune after enumeration.
- **Regression test:** with a pool where the unique optimum requires two moves,
  assert the candidate set includes `n=0,1,2` even when many one-transfer plans
  exist, and that `pick_best_squad` can select it.
- **Severity:** **High** — can miss a free two-move restructure or every viable
  hit, commonly worth 2–8 projected points.

### B6 — rank reranking throws away the multi-gameweek transfer objective

- **Location:** `fpl/pipeline.py:173-202`; `fpl/optimize/rank.py:282-329`.
- **What is wrong:** MILP candidates are generated on discounted horizon xP,
  then the final plan is selected solely by current-event samples. The hit is
  correctly charged in the current event, but none of the candidate's future
  gains are scored. A transfer that loses 0.2 now and gains 8 over the next four
  weeks loses to a one-week move; a hit with strong future payback is nearly
  impossible to choose. Since `rank_sims=4000` by default, this path supersedes
  the otherwise-correct horizon `net_xp` comparison.
- **Concrete fix:** default transfer choice to discounted expected net points
  over a multi-GW state path. If rank utility is explicitly enabled, simulate
  the same horizon, accumulate points and hits, and evaluate final/cumulative
  utility. A current-week rank objective may choose captain and lineup, but must
  not overwrite a five-week transfer decision.
- **Severity:** **High** — directly defeats the purpose of paying a hit only
  when multi-week gain exceeds four; plausible 4–10 point errors.

### B7 — optimizer, simulator, and reported captain are three different decisions

- **Location:** calibrated xP at `fpl/pipeline.py:324-334`; raw rank samples at
  `fpl/pipeline.py:99-110`; rank captain at `fpl/optimize/rank.py:247-267,
  295-303`; reported captain at `fpl/pipeline.py:376-378` and
  `fpl/optimize/lineup.py:55-73`.
- **What is wrong:** after position calibration changes xP, the MILP uses the
  calibrated numbers but `simulate_event` still uses raw rates/minutes. Candidate
  means and simulated means therefore disagree. The rank layer then records its
  own optimal captain in `rank_stats`, but the pipeline discards it and
  `build_lineup` selects a mean/vice-adjusted captain. The displayed team is not
  the team whose `p_beat_target` selected the transfer/squad. Rank scoring also
  omits vice inheritance and autosubs, while production has a vice.
- **Concrete fix:** make one `Decision` object own squad, per-GW XI, bench order,
  captain, vice, chip and samples. Moment-match each player's simulated event
  distribution to the calibrated event mean (or calibrate the distribution
  directly), include vice/autosubs in scenario scoring, and report exactly the
  chosen decision.
- **Severity:** **High** — can change a captain (a doubled outcome) or choose a
  squad on statistics that do not describe the delivered lineup; 2–10 points in
  an affected week.

### B8 — one fixed XI and bench order are used for the whole projection horizon

- **Location:** `fpl/optimize/squad.py:118-151`;
  `fpl/optimize/transfers.py:77-108`; only captain variables are event-specific
  in `fpl/optimize/objective.py:127-151,178-210`.
- **What is wrong:** `start[i]` and bench-slot variables have no event index. The
  solver forces one XI/formation to serve all five gameweeks while allowing the
  captain to change. It undervalues a 15 that rotates cheap defenders/keepers
  around fixtures and can bench a player in this week's report because he was
  more valuable as a horizon-long starter. `build_lineup` reuses that fixed XI
  instead of optimizing `xp_next1`.
- **Concrete fix:** introduce `start[i,g]`, captain/vice and bench order by event,
  with formation constraints per event. At minimum, optimize ownership on the
  horizon and run a separate exact one-week lineup solve for the report.
- **Severity:** **High** — rotation is a core source of squad value; typically
  1–5 points over five weeks, more across blanks/doubles.

### B9 — unavailable players have zero xP but a 35% simulated cameo chance

- **Location:** `fpl/model/minutes.py:216-250`;
  `fpl/model/simulate.py:84-99`.
- **What is wrong:** unavailable status sets `p_start=0`, after which generic
  logic sets `p_play=0.35`; a special case sets `e_minutes=0`, so deterministic
  xP returns zero. The simulator ignores `e_minutes` and draws the player as a
  substitute 35% of the time. This violates the module's asserted mean agreement
  and contaminates shared rank scenarios.
- **Concrete fix:** availability must cap `p_play` directly. For unavailable,
  set `p_start=p_play=p_60=e_minutes=0`; for doubtful, apply availability to both
  the start and cameo branches. Add an all-zero simulation test.
- **Severity:** **Medium** — usually the optimizer avoids zero-xP players, but an
  owned injured starter/rival field can distort captain, autosub and rank results.

### B10 — calibration's intercept is scaled by point ratios, not fixture counts

- **Location:** `fpl/model/calibration.py:109-144`.
- **What is wrong:** `windows = projection / xp_next1` treats “twice this week's
  xP” as “two matches.” An easy future single fixture can receive multiple
  intercepts; a hard one receives a fraction. If `xp_next1` is zero (blank or
  current injury), every column gets exactly one intercept — including the zero
  current event, which can turn a genuine blank into positive xP when the fitted
  intercept is positive. Clipping also means calibrated `xp_next5` need not equal
  the sum of calibrated `xp_gw*` columns, so reporting, optimization and captain
  values can diverge.
- **Concrete fix:** calibrate every `xp_gw{event}` as one event observation,
  using explicit fixture count if the calibration is per fixture; then recompute
  `xp_next1`, raw horizon total and discounted horizon from those calibrated
  columns. A safer small-sample fallback is slope-only calibration.
- **Regression tests:** calibrated horizon equals the sum/discounted sum of
  calibrated event columns for a blank current GW, a DGW, and unequal fixture
  difficulty.
- **Severity:** **High** once calibration activates — it changes positional value
  and therefore budget allocation, especially around blanks/doubles.

### B11 — double-gameweeks overstate current-role evidence

- **Location:** `fpl/data/normalize.py:152-173`;
  `fpl/model/minutes.py:155-189`.
- **What is wrong:** `gws_played` counts distinct rounds, while `starts` sums
  fixture rows. In a DGW a player can have two starts and one “opportunity,” so
  `p_start` can exceed one before clipping and current form gets the wrong sample
  size. The comment claims the field counts matches, but the implementation
  counts gameweeks.
- **Concrete fix:** store both `matches_available`/fixture rows and
  `gameweeks_observed`. Use matches for start probability and exposure minutes;
  retain gameweeks only for horizon/form timing.
- **Severity:** **Medium** — most visible after doubles; it can turn a rotation
  risk into a falsely nailed player for subsequent weeks.

### B12 — Tier 1 “walk-forward” validation leaks the target distribution

- **Location:** `fpl/backtest/aggregate.py:15-44`.
- **What is wrong:** only each player's final season is predicted, not every
  eligible season. More importantly, `pop_mean` is computed from the entire
  frame, including target seasons, and the “naive” baseline is the mean of the
  target actuals. Both are future-informed.
- **Concrete fix:** iterate season cutoffs. For target season `t`, construct all
  player histories and positional priors only from seasons `< t`; score all
  eligible players in `t`; make the naive baseline a pre-`t` population or prior
  season estimate. Report metrics by cutoff and pooled out-of-sample.
- **Severity:** **Critical validation risk** — no direct point loss, but it can
  approve a worse rate model and affect every weekly decision.

### B13 — the production walk-forward test uses future data and a free Wildcard every week

- **Location:** the contamination is admitted at
  `fpl/backtest/walkforward.py:15-24`; implementation at
  `scripts/run_walkforward.py:74-90,110-143`.
- **What is wrong:** historical GWs use the newest bootstrap's price, team,
  ownership, status, news and set-piece role. A September injury is known in an
  August replay; transferred players are mapped to their later club. Every week
  then calls `optimize_squad` from scratch, not the transfer optimizer, with
  `rank_sims=0`. It scores a synthetic captain rather than the production rank
  captain, and compares a free weekly rebuild without chips/hits against an FPL
  field average that contains real state, hits and chips. This is an oracle upper
  bound, not a production backtest.
- **Concrete fix:** archive point-in-time bootstrap, fixtures, summaries and
  deadlines. Replay one manager state sequentially from GW1: permanent squad,
  purchase/selling prices, bank, FTs, transfers, hits, per-GW lineup/captain/vice,
  chips and FH restoration. Run expected-points and rank policies side by side
  against hold, FPL xP, and a simple transfer baseline. Keep the scratch rebuild
  only as a clearly labelled oracle ceiling.
- **Severity:** **Critical validation risk** — it can create confidence in a
  policy that is impossible to execute.

### B14 — mutable ledger entries can train calibration on post-deadline/replayed forecasts

- **Location:** `fpl/backtest/ledger.py:50-80,104-112`;
  `fpl/model/calibration.py:147-176`; unconditional production write at
  `fpl/pipeline.py:336-341`.
- **What is wrong:** immutable versions exist, but `gwN.parquet` is overwritten
  on every ordinary pipeline run and `load_predictions` always reads that
  mutable file. Nothing marks the forecast actually acted on or chooses the last
  version before the deadline. A post-deadline run or replay can therefore
  replace the forecast scored and used for later calibration. `MODEL_VERSION`
  is also a hand-maintained date that predates current core commits.
- **Concrete fix:** write an append-only manifest with `origin`, `created_at`,
  deadline, model/config/data hashes, and `actioned_at`/`confirmed`. Scoring and
  calibration must select the confirmed actioned version, otherwise the latest
  version strictly before deadline, and must reject `origin=replay`. Derive model
  version from Git SHA plus dirty flag.
- **Severity:** **High** — feedback leakage can bias all later forecasts and erase
  the only honest live evidence.

### B15 — Tier 2 uses actual minutes and excludes DNPs while powering a production trust gate

- **Location:** `scripts/run_backtest.py:97-131`;
  claim in `fpl/backtest/gw_level.py:1-4`.
- **What is wrong:** rows are filtered to `minutes > 0`, then predictions are
  multiplied by the minutes that actually occurred. This gives the rate model
  future playing time and removes every nonappearance, the main source of FPL
  error. It cannot validate minutes, captaincy, fixture-level xP, or the end-to-end
  pipeline, despite the “trust gate” language. Separately,
  `fpl/backtest/gw_level.py:38-48` calls a captaincy “hit” when the globally
  highest-projected player is also the globally highest actual scorer; it does
  not evaluate the captain chosen from the manager's owned XI.
- **Concrete fix:** retain all registered player-GW rows, use only point-in-time
  expected-minute distributions, score full xP, and treat the current rate-only
  test as a separately named component diagnostic that cannot make the production
  trust decision.
- **Severity:** **High validation risk** — likely flatters the model precisely
  where weekly selection is hardest.

### B16 — a partial element-summary fetch is indistinguishable from a genuine newcomer

- **Location:** `fpl/data/client.py:115-142`;
  `fpl/data/normalize.py:71-101`.
- **What is wrong:** individual failures are silently omitted; missing prior rows
  are then zeroed and routed to the same price prior as a new signing. A partial
  API outage can erase an established player's history while the optimizer still
  returns a confident legal team. `client.stale` is only a global warning.
- **Concrete fix:** return per-player fetch status, retain the newest cached row
  for failed IDs, and enforce a coverage gate before optimization (for example,
  abort if any owned player or any serious candidate is missing, or if overall
  coverage falls below 99%). Distinguish `new_to_league`, `no_history`, and
  `fetch_failed` in the normalized contract.
- **Severity:** **High** during an outage — one corrupted premium estimate can
  trigger a bad sale/hit; normally dormant.

### B17 — the strict club-cap constraint misses FPL's real-transfer exception

- **Location:** `fpl/optimize/squad.py:140-143`;
  `fpl/optimize/transfers.py:94-108`.
- **What is wrong:** a real Premier League transfer can temporarily leave an FPL
  manager with four players from one club. FPL allows the squad to remain and
  requires it to return to three only when the manager next makes a transfer.
  The zero-transfer hold is infeasible in this code, so it can force a move that
  the game does not require.
- **Concrete fix:** in transfer mode, permit the current club overage only for
  `n=0`; for any transfer plan enforce the ordinary cap. A new/rebuilt/WC/FH
  squad remains capped at three.
- **Severity:** **Low/Medium** — rare, but can force an unnecessary transfer or
  hide a legal hold.

## 2. Robustness gaps in the current approach

### R1 — the model optimizes a one-week median-beat event, not expected season rank

- **Location:** `fpl/optimize/rank.py:230-244,309-329`;
  defaults at `fpl/config.py:45-60`.
- **The model does X, which breaks down when Y:** it maximizes
  `P(this GW score > the simulated field median)` at the default target of 0.5.
  That is neither expected points, expected rank percentile, nor probability of
  reaching a season/mini-league goal. It breaks down whenever current points,
  remaining weeks, rank tier, or the size of the deficit matters. A team can
  have lower mean and expected rank but a slightly better chance to cross one
  weekly threshold. Strict `>` also gives no credit for ties, which particularly
  penalizes template-heavy outcomes even though tied classic-league teams share
  position after the transfer-count tiebreak.
- **Proposed change:** make discounted expected total points the default primary
  objective. Offer rank utility only with an explicit stateful target: current
  points/rank or mini-league deficit, weeks remaining, and a target cohort. A
  useful approximation is `P(cumulative gain over target cohort >= deficit)`;
  the full version simulates final rank utility. Report mean points, expected
  percentile, downside and target probability rather than hiding them behind one
  scalar.
- **Severity:** **High** — this is the final selector in the default weekly path,
  so an ill-posed target can repeatedly trade points for the wrong kind of risk.

### R2 — the field model is not the field the user is trying to beat

- **Location:** `fpl/optimize/rank.py:144-206`.
- **The model does X, which breaks down when Y:** it samples a uniformly random
  legal formation from overall ownership × model xP, ignores 15-player budget,
  club cap, benches, autosubs, vice-captaincy and observed captain shares, then
  captains the highest model-xP player. It breaks down for a high-owned premium
  versus a low-owned differential because overall ownership is not effective
  ownership in the user's rank cohort. It can create impossible collections of
  premiums and overstate field strength/correlation.
- **Proposed change:** collect pre-deadline public picks for a stratified sample
  around the user's overall-rank/mini-league cohort. Simulate the actual 15,
  lineup, captain, vice and autosubs. Until enough data exists, calibrate formation
  and captain shares empirically and enforce budget/club legality in generated
  rivals. Backtest candidate rankings against actual cohort outcomes.
- **Severity:** **High** for rank mode; no effect when rank simulation is disabled.

### R3 — the joint simulation permits impossible match outcomes

- **Location:** `fpl/model/simulate.py:69-75,101-124`.
- **The model does X, which breaks down when Y:** each team's defenders share a
  goals-conceded draw, but the opponent's attackers draw goals independently.
  An attacker can score in a scenario where opposing defenders also receive a
  clean sheet. Player goals and assists are independent rather than allocated
  from a team score, and every player's bonus is an independent binomial rather
  than a within-match BPS ranking. It breaks down exactly where rank utility is
  supposed to add value: stacks, opposing players, captain ceilings and bonus
  covariance.
- **Proposed change:** simulate each fixture once: joint home/away scoreline,
  coherent lineups/minutes, allocate team goals and assists to on-pitch players,
  derive clean sheets/conceded from the same score, then calculate BPS and award
  3/2/1 with tie rules. Preserve marginal means by calibrating allocation shares.
- **Severity:** **High** — expected xP may remain close, but covariance and tails
  driving the rank decision are materially wrong.

### R4 — confidence labels never affect the projection distribution or decision

- **Location:** confidence produced at `fpl/model/minutes.py:150-204` and
  `fpl/model/strength.py:144-157`; consumed only as display fields in
  `fpl/model/xp.py:190-194`.
- **The model does X, which breaks down when Y:** it emits the same precise scalar
  currency for an established player and a price-prior newcomer. `p_start`,
  per-90 rates, team strength, calibration coefficients and manual overrides are
  treated as known parameters in simulation. Match randomness is sampled;
  epistemic uncertainty about the inputs is not. It breaks down around new
  signings, role changes, promoted teams and stale team news, where false
  precision invites brittle transfers.
- **Proposed change:** retain posterior/sample-size parameters: beta draws for
  start/60 probabilities, gamma or hierarchical draws for rates, and uncertainty
  bands for team strengths/calibration. Generate posterior-predictive scenarios.
  For irreversible moves/hits, show `P(net gain > 0)` and a credible interval in
  addition to expected gain. Do not blindly subtract an “uncertainty penalty”
  from every low-confidence player; use it where utility is asymmetric or as a
  near-tie guardrail.
- **Severity:** **High** — low-signal estimates can drive premium transfers and
  hits as confidently as stable ones.

### R5 — minutes are current-week, independent per player, and fixed across the horizon

- **Location:** `fpl/model/minutes.py:126-266`;
  `fpl/model/xp.py:149-199`; `fpl/model/simulate.py:84-99`;
  known limitation in `fpl/data/overrides.py:3-18`.
- **The model does X, which breaks down when Y:** one `p_start/p_play/m_start`
  profile is reused for every future event, and player starts are independently
  drawn. It breaks down for injury return dates, suspensions, cup rotation,
  fixture congestion, manager rotation patterns, and the deputy whose minutes
  rise when a teammate is out. Simulated teams can have more than eleven starters
  and incoherent total minutes.
- **Proposed change:** produce `(player,event)` minute distributions with
  availability windows and congestion/cup indicators. Reconcile starts within
  each team/position group to eleven starters and roughly 990 regulation minutes,
  using a lightweight depth chart and substitution shares. An injury should
  remove minutes from one player and reallocate them to named alternatives.
  Manual overrides should target an event range and expire hard when stale.
- **Severity:** **High** — minutes is already the strongest measured part of the
  model; structural role misses dominate one-week xP.

### R6 — cold starts, missing data and stale overrides collapse to deceptively simple priors

- **Location:** `fpl/data/normalize.py:71-101`;
  `fpl/model/minutes.py:172-204`; `fpl/model/scoring.py:19-34,140-190`;
  `fpl/data/overrides.py:53-101`; current entries at
  `data/overrides.yaml:17-49`.
- **The model does X, which breaks down when Y:** no prior PL row becomes zero
  history, a price-derived start prior and positional scoring mean. Current form
  is blended by gameweek count rather than effective minutes, so a tiny sample
  can receive up to 60% weight. Positional priors are unweighted means of player
  per-90 rates, giving a one-minute player the same influence as a 3,000-minute
  player. Stale overrides are warned about but still applied at full configured
  weight; the two checked in August are still active until GW6 despite a 48-hour
  freshness budget.
- **Proposed change:** use exposure-weighted/hierarchical positional priors;
  blend current rates by effective minutes/opportunities, not only round count;
  ingest a modest cross-league prior for newcomers (league-strength-adjusted
  xG/xA, previous minutes/starts and role) or require a manual cold-start input.
  Separate missing-fetch from newcomer. Decay stale overrides toward the model
  or refuse optimization until owned-player news is refreshed.
- **Severity:** **High** early season and after transfers; the documented Tzolis
  case remains only partly fixed because the override changes minutes, not talent.

### R7 — attacking fixture adjustment likely counts team quality twice

- **Location:** rates at `fpl/model/scoring.py:19-34`;
  multiplier at `fpl/model/fixtures.py:30-42`; applied at
  `fpl/model/xp.py:128-130`.
- **The model does X, which breaks down when Y:** historical individual xG/xA per
  90 already reflects the player's attacking environment, then the forecast
  multiplies by the current club's attack rating again, as well as opponent
  defence and venue. For a player remaining at a strong club this can double
  count club quality; for a transferred player it has no origin-club denominator.
- **Proposed change:** either (a) use only opponent/venue multipliers on observed
  individual rates, or (b) decompose team expected goals and player shares:
  project a match-level team goal/xG total from both teams and venue, then allocate
  it by the player's normalized xG/xA share. For transfers, carry the origin-team
  context used to normalize the prior. Validate both variants prospectively.
- **Severity:** **High** if the multiplier is materially dispersed — it can
  systematically overstack elite attacks and misprice transfers between clubs.

### R8 — transfer logic prices the hit, but not the future transfer path or option value

- **Location:** `fpl/optimize/transfers.py:64-143,210-235`;
  decay heuristic at `fpl/config.py:28`.
- **The model does X, which breaks down when Y:** it chooses one move now and
  assumes the resulting fixed squad/XI is held for the whole horizon. Future
  free transfers, bank, selling values and future buys do not exist. A small
  positive gain spends an FT with no value assigned to rolling it; a gain in GW5
  is credited now even if the move could be delayed; expected price changes have
  no effect on future feasibility. Current selling-value and bank constraints
  are correct, but only at today's prices.
- **Proposed change:** implement a rolling 4–6 GW multi-period MILP with
  ownership, starts, transfers in/out, bank, purchase/selling value approximation,
  FTs and hits by event. Add a terminal salvage value for banked FT and bank/team
  value, estimated by backtest rather than guessed. A lightweight interim fix is
  a configurable value for `FT_next` plus a “wait one week” candidate and
  price-change scenario bands. Price chasing should remain a tie-breaker unless
  it changes a future feasible move.
- **Severity:** **High** — repeated sideways transfers and early moves can lose
  several points plus flexibility even when each isolated horizon delta is
  slightly positive.

### R9 — chip timing beyond the horizon uses structure, not chip value

- **Location:** `fpl/pipeline.py:308-315`;
  `fpl/optimize/chips.py:77-153,180-265`.
- **The model does X, which breaks down when Y:** past the xP horizon it counts
  blanks/doubles for the current 15 all the way to GW38. It breaks down because
  first-half chips expire at GW19, unscheduled postponements have no event, the
  future optimal squad is not the current 15, and fixture count is not expected
  chip gain. Within the horizon, BB value is the four numerically lowest players,
  which can be an illegal bench under formation constraints; TC accepts any
  captain DGW even below its xP threshold; FH counts blanks among all 15 rather
  than marginal score of an optimized FH XI; and hard-coded priority FH → WC →
  TC → BB never compares opportunity value.
- **Proposed change:** use each chip's legal window and compare marginal points
  from complete optimized actions against no-chip action under fixture scenarios.
  Enumerate plausible chip weeks/scenarios rather than pretending unscheduled
  fixtures are known. Run a legal per-event lineup solve for BB, optimize a real
  one-week FH squad, and value TC as the extra captain return including captain
  and vice nonappearance scenarios. Replan weekly; do not solve all 38 weeks with
  false precision.
- **Severity:** **High** — chip timing and chip squad quality are worth much more
  than the fine differences between ordinary transfers.

### R10 — goalkeeper/bonus bias remains structural, not just a calibration offset

- **Location:** `fpl/model/bps.py:13-33`;
  `fpl/model/scoring.py:10-16`; `fpl/model/simulate.py:120-124`.
- **The model does X, which breaks down when Y:** it carries realized historical
  `bonus90` forward, mildly fixture-scales it, and draws bonus independently per
  player. It cannot represent the changed 2026/27 BPS competition or the fact
  that only the top match BPS scores receive 3/2/1. STATUS's measured goalkeeper
  bias (+0.60, then +0.72) remains. Position calibration can hide average bias
  without fixing player ordering or covariance.
- **Proposed change:** rebuild projected BPS from the events already simulated,
  add the current rule table, rank all match participants jointly with tie rules,
  and score GK save/bonus calibration separately. Until then, cap confidence in
  goalkeeper and bonus components and compare to a no-bonus challenger.
- **Severity:** **Medium/High** — repeated budget and captain decisions can be
  biased even if total mean calibration looks acceptable.

### R11 — the default squad builder silently optimizes for Bench Boost readiness

- **Location:** hidden default at `fpl/config.py:31-39,91-92`; enforced at
  `fpl/optimize/squad.py:144-151`; absent from `config.yaml`.
- **The model does X, which breaks down when Y:** every fresh build requires all
  four bench players to have at least 2.5 current-week xP, even when BB is spent
  or not under consideration. That spends ordinary-XI budget to solve a chip
  readiness problem, and users cannot see the setting in the checked-in config.
- **Proposed change:** set the general default to zero; enable a bench floor only
  in WC/BB preparation scenarios and expose it in the report with its XI cost
  and bench gain. Prefer an explicit BB marginal-value objective to a hard floor.
- **Severity:** **Medium** — measured cost was small in one GW, but it can become
  material with expensive premiums or after BB has been used.

## 3. Ideas from comparable models and adjacent fields

### Adopt now — a short-horizon multi-period MILP

- **Source:** [open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver)
  uses player/gameweek ownership, lineup, captain, vice, bench, transfer, bank,
  FT and chip variables; [AIrsenal](https://pypi.org/project/airsenal/) predicts
  and optimizes a transfer strategy over a deliberately short three-week horizon.
- **Problem in this repo:** R8/B8 — one current transfer and one fixed XI cannot
  value rolling FTs, delayed transfers, rotation or future bank.
- **Where it goes:** replace/extend `fpl/optimize/transfers.py` with indexed
  `owned[p,g]`, `start[p,g]`, `tin/tout[p,g]`, `bank[g]`, `ft[g]`, `hit[g]`,
  captain/vice and bench-slot variables. Reuse current squad legality constraints
  per gameweek. Keep a 4–6 GW rolling horizon with terminal FT/bank salvage.
- **Worth the complexity?** **Yes.** This is the highest-value architectural
  improvement after correctness fixes, and still modest for one weekly user.
  Do not optimize all 38 weeks; false long-range precision and solve complexity
  add little.

### Adopt next — one coherent match simulation, then allocate player events

- **Source:** AIrsenal separates team-level score prediction and player-level
  contribution; the existing repo's own rank rationale requires joint outcomes.
  FPL-focused xMins systems also use scenario-weighted minutes rather than one
  scalar ([FPL Review xMins methodology](https://docs.fplreview.com/the-model/projections/xmins/)).
- **Problem in this repo:** R3/R5 — impossible attacker-versus-clean-sheet
  outcomes, unconstrained team minutes, independent bonus and missing deputies.
- **Where it goes:** `fpl/model/simulate.py`, supported by event-indexed output
  from `minutes.py`, a team goal layer in `strength.py/fixtures.py`, and rebuilt
  BPS in `bps.py`.
- **Worth the complexity?** **Yes, staged.** First make scoreline and clean sheets
  coherent; then goal/assist allocation; then BPS. A full proprietary news model
  is not appropriate for one user, but a small depth chart plus editable,
  expiring overrides is.

### Adopt — prospective challenger models and proper distribution scoring

- **Sources:** [OpenFPL](https://arxiv.org/abs/2508.09992) reports
  position-specific public-data ensembles trained on four seasons and tested
  prospectively on the following season across one-, two- and three-GW horizons.
  Gneiting and Raftery's [proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
  explain why probabilistic forecasts should be judged for calibration and
  sharpness, not only point MAE/rank correlation.
- **Problem in this repo:** confidence is unvalidated; current metrics cannot
  tell whether 70% start/CS probabilities occur about 70% of the time; current
  backtests are not point-in-time.
- **Where it goes:** point-in-time `fpl/backtest/` datasets and a challenger
  interface beside `scoring.py/minutes.py`. Add Brier/log score and reliability
  bins for start, play, 60 and clean sheet; CRPS or ranked probability score for
  player points; decision metrics for squad/transfer regret.
- **Worth the complexity?** **Yes for evaluation, later for ML.** Proper scores
  are cheap. Do not replace the transparent model with XGBoost/ensembles until
  the data ledger is point-in-time and the simple baseline is honestly beaten.

### Adopt selectively — DFS threshold utility, but only for a real chase target

- **Source:** Hunter, Vielma and Zaman's
  [DFS integer-programming work](https://arxiv.org/abs/1604.01455) maximizes the
  chance that at least one entry wins a top-heavy contest, using expected score,
  variance and correlation across a portfolio of entries.
- **Problem in this repo:** `rank.py` borrows the “top-heavy tournament” idea
  without the key context: this user has one persistent entry and the configured
  threshold is not linked to a season prize, rank or mini-league deficit.
- **Where it goes:** an optional stateful utility in `rank.py`: simulate
  cumulative differential points against a chosen cohort/leader and maximize
  probability of closing the known gap. Continue using shared scenarios to value
  ownership correlation.
- **Worth the complexity?** **Only late-season or for a defined mini-league
  target.** Explicitly skip multi-entry portfolio diversification and generic
  “force high variance” constraints: there is only one FPL team, no lineup
  portfolio, and no weekly winner-take-all payout in the normal use case.

### Borrow the diagnostic, skip Markowitz as the main objective

- **Source:** Markowitz's [mean-variance portfolio framework](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)
  highlights covariance rather than standalone variance.
- **Problem in this repo:** defensive stacks, attacking stacks and opposing
  players have correlated returns, which raw summed xP cannot show.
- **Where it goes:** use scenario covariance in the report: expected points,
  standard deviation, downside quantile, and contribution of club stacks. It can
  also generate diverse candidate squads near the xP optimum.
- **Worth the complexity?** **As a diagnostic, yes; as `mean - lambda*variance`,
  no.** FPL utility is not quadratic, covariance estimates are unstable, and the
  existing scenarios can evaluate the actual nonlinear target without reducing
  it to one arbitrary risk-aversion coefficient.

### Explicitly skip Kelly sizing; keep only its “edge must clear uncertainty” lesson

- **Source:** Kelly's [original growth-rate criterion](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x)
  sizes repeatable wagers to maximize logarithmic bankroll growth.
- **Problem in this repo:** there is no divisible bankroll or fractional transfer;
  a move is discrete, hits cost fixed FPL points, and rank/points are not
  multiplicative wealth.
- **Where it goes:** nowhere as an objective. The useful analogue is reporting
  posterior `P(multi-GW gain > hit)` and being conservative on thin edges, which
  belongs in `transfers.py` decision diagnostics.
- **Worth the complexity?** **No Kelly implementation.** It would add impressive
  terminology without matching the game's decision structure.

### Adopt — immutable actioned forecasts and an executable manager-state replay

- **Source:** this is standard prospective forecast evaluation, and AIrsenal's
  optimizer explicitly works from transaction state over a chosen short horizon.
- **Problem in this repo:** B12–B15 — the current harness tests an oracle free
  rebuild on future-contaminated inputs and may score a replaced ledger entry.
- **Where it goes:** `fpl/data/` snapshot manifests,
  `fpl/backtest/ledger.py`, and a new sequential policy replay in
  `fpl/backtest/walkforward.py`.
- **Worth the complexity?** **Essential.** For one weekly user, a trustworthy
  backtest is more valuable than another forecasting feature because it tells
  whether any feature should influence a transfer.

## Recommended implementation order for Claude Code

### P0 — make actions legal and state reversible

1. Implement half-season chip inventory and legality from `chip_events`.
2. Correct WC/FH FT carry and replace the wrong existing tests.
3. Preserve and restore the permanent FH squad/bank/purchase prices.
4. Require explicit chip confirmation; make WC/FH return their actual optimized
   squad or demote advice to a non-actionable timing flag.
5. Stratify transfer candidates by transfer count.

**Acceptance gate:** synthetic tests covering all eight chips, GW19/20, one-chip
per GW, FT balances 0–5, FH restoration, and a two-transfer optimum. No live run
may return a chip without a complete legal action.

### P1 — make validation honest before tuning the model

1. Add immutable `actioned` forecast selection and exclude replay forecasts from
   calibration.
2. Repair aggregate cutoff leakage.
3. Build point-in-time snapshot manifests and a sequential manager-state replay.
4. Score production lineup, vice/autosubs, hits and chips; A/B rank mode against
   expected-points mode and simple baselines.

**Acceptance gate:** a replay can prove every datum was timestamped before its
deadline and reproduce squad/bank/FT/chip state from GW to GW. Scratch rebuild is
reported only as an oracle ceiling.

### P2 — align objective and simulation

1. Keep expected multi-GW net points as default; stop one-week rank reranking of
   transfer plans.
2. Add per-event lineup variables or a separate exact weekly lineup solve.
3. Make calibrated event means and sample means agree; return/report the exact
   rank-selected captain/vice.
4. Replace independent team-side draws with coherent match scenarios.
5. Calibrate the rival field from actual rank-cohort picks before allowing rank
   utility to be primary.

**Acceptance gate:** for every candidate, deterministic event xP equals simulated
mean within Monte Carlo tolerance; reported decision equals scored decision; no
scenario contains a goal against a clean sheet in the same match.

### P3 — improve information, then sophistication

1. Event-specific team-coherent minutes and injury redistribution.
2. Exposure-weighted/hierarchical rate priors and newcomer inputs.
3. Team-goal/player-share attack model and projected BPS.
4. Rolling multi-period MILP with FT/bank terminal value and chip scenarios.
5. Proper probability scoring and prospective challenger models.

**Acceptance gate:** every new component beats a frozen simpler challenger on
point-in-time out-of-sample data and improves decision regret, not merely in-sample
fit or one gameweek's score.
