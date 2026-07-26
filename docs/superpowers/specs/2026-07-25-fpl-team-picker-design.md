# FPL Team Picker — Design Spec

**Date:** 2026-07-25
**Status:** Approved for planning
**Target:** usable GW1 squad recommendation before the 2026/27 GW1 deadline, Fri 21 Aug 2026 18:30 UK (`2026-08-21T17:30:00Z`)

## 1. Purpose and scope

A Python decision-support system that recommends an FPL squad, starting XI, captain, and transfers from data rather than intuition.

**It never writes.** No transfers, chips, or lineup changes are submitted. Output is a report the user applies manually on the FPL site. No endpoint or tool requiring a login/session cookie is called.

First run is **Mode 1** (build 15 players from scratch under £100.0m). The user has no entry ID yet. Mode 2 (weekly transfers) is built in this pass but stays unexercised until a team ID exists.

## 2. Verified environment (Phase 0, measured 2026-07-25)

### MCP server
`fantasy-pl` (`python -m fpl_mcp`), **16 tools, all read-only** — no transfer, chip, or lineup write tool exists. Configured at **user scope** in `~/.claude.json`; the project has no `.mcp.json`. The three skills are project-scoped in `.claude/skills/`.

- **No ID required (9):** `get_gameweek_status`, `analyze_players`, `search_fpl_players`, `get_player_information`, `analyze_fixtures`, `analyze_player_fixtures`, `get_blank_gameweeks`, `get_double_gameweeks`, `compare_players`
- **Entry/team ID required (5):** `get_team`, `get_manager`, `get_manager_info`, `get_league_standings`, `get_league_analytics`
- **OFF-LIMITS — credential-dependent (2):** `get_my_team`, `check_fpl_authentication`. Read-only but consume stored FPL login credentials. Never called. (Server also reports `No credentials found in any source`.) Mode 2 uses public `get_team(team_id)` / `entry/{id}/event/{gw}/picks/` instead.

### Season state — the dominant design constraint
The API has rolled over to 2026/27 with **zero gameweeks played**. Every GW-level endpoint is empty:

| Endpoint | Measured 2026-07-25 |
|---|---|
| `bootstrap-static/` events | 38 events, `finished=0`, all `average_entry_score=0` |
| `element-summary/{id}/` `history` | 0 rows (verified across player IDs 12, 1, 200, 400, 550) |
| `event/{1,10,38}/live/` | `elements=0` |
| `fixtures/` | 380 fixtures, `finished=0`, 0 populated stats blocks |
| `entry/{id}/...` | Requires a manager ID; never returns player-level GW stats |

**What *is* available:**
- 558 players, 20 teams, 38 events, all 380 fixtures with per-fixture FDR (`team_h_difficulty`, `team_a_difficulty`), 10 fixtures/event through at least GW8 — no blanks or doubles in the early schedule.
- Bootstrap season stats are **2025/26 carryover** (Saka: 157 pts, 2218 mins, 7.57 xG — matches his `history_past` 2025/26 row exactly). A real per-90 baseline exists for GW1.
- `element-summary/{id}/history_past` holds **up to 8 prior seasons** per player (sampled: 8/5/3/2/1 by career length), 30 fields each including `expected_goals`, `expected_assists`, `expected_goals_conceded`, `bps`, `starts`, `minutes`, `defensive_contribution`, `clearances_blocks_interceptions`, `tackles`, `recoveries`.

**What is missing and must be worked around:**
- `form` is `0.0` for all 558 players → the form component of the model **cannot** contribute at GW1.
- `strength_attack_home/away` and `strength_defence_home/away` are **all 0**; `strength` is `None`. Only `strength_overall_home/away` is populated. → team strength must be derived, not read.
- `news` is empty and `chance_of_playing_next_round` is `null` for everyone → GW1 availability depends entirely on web-sourced team news.
- No GW-level history anywhere in the API → the Phase 2 backtest needs an external archive.

### Toolchain
Python 3.13.1 with `pulp` 2.9.0, pandas 2.2.1, numpy 1.26.4, scipy 1.14.1, scikit-learn 1.6.0, pyarrow 18.1.0, requests, httpx. Nothing to install.

## 3. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Data path | Direct HTTP for bulk snapshot; MCP for enrichment | `analyze_players` is `limit`-capped curated JSON over a subprocess; `bootstrap-static/` + `fixtures/` is what the MCP server itself wraps. Keeps the headless path MCP-free by construction. |
| MCP's role | Blanks/doubles detection, fixture analysis, interactive player verification | Genuinely additive; never on the critical path |
| Backtest | Both tiers: API multi-season aggregates + vaastav per-GW CSVs | No FPL endpoint can validate fixture adjustment, form decay, or captaincy pre-season |
| Risk profile | Balanced — pure xP, no ownership tilt | Lands on template where template is optimal, differential where it isn't |
| Archive scope | `backtest/archive.py` only, never in the weekly path | Cold-start crutch. From GW2 the user's own cached `element-summary/history` and `event/{gw}/live/` become the GW-level source |

## 4. Architecture

```
FPL Team Picker/
├── run_gameweek.py          # single entry point — pure Python, no MCP, no skills
├── config.yaml
├── requirements.txt
├── fpl/
│   ├── config.py            # load + validate config.yaml
│   ├── state.py             # persisted FT balance + chip usage → data/state.json
│   ├── data/
│   │   ├── client.py        # HTTP: bootstrap-static, fixtures, element-summary, entry/*
│   │   ├── cache.py         # snapshots, freshness, retention
│   │   ├── normalize.py     # ID→name, now_cost/10, status+news flags → DataFrames
│   │   ├── store.py         # parquet persistence
│   │   └── archive.py       # vaastav CSV loader (backtest only)
│   ├── model/
│   │   ├── strength.py      # derived team attack/defence ratings
│   │   ├── minutes.py       # p_start, p_60, availability
│   │   ├── scoring.py       # points-per-90 by position, EW form blend
│   │   ├── fixtures.py      # FDR + home/away adjustment, DGW/BGW detection
│   │   ├── bps.py           # bonus tendency
│   │   └── xp.py            # assembles xP for next 1 and next 5 GWs
│   ├── optimize/
│   │   ├── squad.py         # Mode 1 MILP
│   │   ├── transfers.py     # Mode 2 MILP
│   │   ├── lineup.py        # XI / formation / bench order / captain
│   │   └── chips.py         # chip heuristics
│   ├── backtest/
│   │   ├── aggregate.py     # history_past multi-season validation
│   │   └── gw_level.py      # per-GW backtest + error metrics
│   └── report/weekly.py     # weekly-report format
├── data/{cache,processed,archive}/
└── tests/
```

### Module contracts (the swap seams)

The two boundaries that matter for the user's "swappable model or optimizer" requirement:

**`model/xp.py` → `optimize/`** — one DataFrame, fixed columns:

| Column | Meaning |
|---|---|
| `player_id`, `web_name`, `team`, `position`, `price` | identity |
| `xp_next1` | expected points, next event (sums fixtures — DGW-aware, 0 for a blank) |
| `xp_next5` | expected points over next 5 events |
| `p_start`, `e_minutes` | minutes model output |
| `confidence` | `high` / `medium` / `low` |
| `flags` | list of human-readable uncertainty strings, carried through to the report |

`optimize/` reads **only** these columns. Any model producing this frame is a drop-in replacement.

**`optimize/` → `report/weekly.py`** — a `Recommendation` object (squad 15, XI, bench order, captain, vice, transfers with net xP, chip advice, budget). The report **never recomputes anything**; it formats.

## 5. Data layer

Per the `fpl-data-fetch` skill, which is authoritative:

- **Endpoints:** `bootstrap-static/`, `fixtures/` (+ `?event={gw}`), `element-summary/{id}/`, `entry/{id}/`, `entry/{id}/history/`, `entry/{id}/event/{gw}/picks/`. None require auth. `my-team/` is never touched.
- **Freshness:** reuse a cached snapshot < 6h old, or < 1h old on matchdays. Skip the fetch when fresh. "Matchday" is defined concretely as: any UTC date on which at least one fixture has a `kickoff_time`, read from the cached fixture list — not a hardcoded weekday.
- **Snapshots:** raw JSON at `data/cache/{endpoint_slug}_{ISO8601}.json`, keep the **last 3** per endpoint, delete older.
- **Processed:** parquet in `data/processed/` — one table per entity: `players`, `teams`, `fixtures`, `player_gw_history`, `my_squad`.
- **Rate limit:** ≤ ~1 req/sec; `element-summary` calls sequential with a small delay. 558 sequential calls ≈ 10 min, so a full per-player sweep is opt-in (`--deep`), not every run. The GW1 build needs only `bootstrap-static/` + `fixtures/` plus `element-summary` for the shortlist.
- **Normalization:** resolve team and `element_type` IDs to names before anything downstream sees them; `now_cost / 10` → £m; flag `status != 'a'` (`d` doubtful, `i` injured, `s` suspended, `u` unavailable) and carry `news` through.
- **Failure:** on request failure or API downtime, fall back to the newest cached snapshot and state in the report that data may be stale. Never block the pipeline.

**Supplementary signal** (xG/xA trends, press-conference team news, price predictions) via web search/fetch, per the same skill: always cite the source, never fabricate or estimate a missing number — report it as missing.

## 6. xP model

`xP` is assembled per player per fixture, then summed across the fixtures in an event (which makes DGW/BGW handling fall out for free).

### Components
Points weights are position-dependent per FPL scoring:

| Component | Formula |
|---|---|
| Appearance | `p_play + p_60` (1 pt for any minutes, 2 for 60+) |
| Goals | `xG90 × (E[mins]/90) × goal_pts[pos]` — GK/DEF 6, MID 5, FWD 4 |
| Assists | `xA90 × (E[mins]/90) × 3` |
| Clean sheet | `p_CS × cs_pts[pos] × p_60` — GK/DEF 4, MID 1, FWD 0 |
| Goals conceded | GK/DEF only: `−0.5 × E[GC]` (−1 per 2 conceded) |
| Saves | GK only: `saves90 × (E[mins]/90) / 3` |
| Defensive contribution | `p(threshold) × 2` — **direct points**, DEF 10+ CBIT, MID/FWD 12+ CBIRT (see §6.6) |
| Bonus | expected bonus from shrunk bonus-per-90, fixture-adjusted (see §6.4) |
| Cards / own goals | small negative from last-season per-90 rate |

### 6.0 Two scoring paths from the same raw actions — do not double-count
Clearances, blocks, interceptions, tackles, and recoveries feed **two independent scoring mechanisms**, and conflating them is an easy implementation error:

1. **Direct points** via the Defensive Contribution threshold (a 2025/26 rule change) — owned by the DC component in the table above.
2. **The BPS matrix**, which determines expected 1/2/3 bonus points — owned by `model/bps.py`.

These are additive and both real. `bps.py` must model *bonus only* and must not re-award threshold points; the DC component must not attempt to model bonus. Verified empirically (§6.6).

### 6.1 Baseline with shrinkage
Small-minutes players must not be treated as reliable per-90 samples:

```
pts90_shrunk = (minutes × pts90_player + k × pts90_position_mean) / (minutes + k)
```

`k` defaults to 900 minutes, fitted in the backtest. Applied to every per-90 rate (xG90, xA90, bonus90, defensive contribution).

### 6.2 Form blend
```
pts90 = w_form × EW_pts90(last 6 GWs, half-life h) + (1 − w_form) × pts90_shrunk
w_form = min(1, gws_played / 6) × w_max
```

**At GW1, `gws_played = 0` so `w_form = 0`** — the model is 100% baseline by necessity, not by choice. `h` and `w_max` are fitted on the GW-level backtest rather than guessed; defaults `h = 3` GWs, `w_max = 0.6`.

### 6.3 Minutes model
- `p_start` from last season `starts / team_games_available`, shrunk as in §6.1.
- Hard override on `status`: `i`/`s`/`u` → `p_start = 0`; `d` → scale by `chance_of_playing_next_round` when present.
- Team news from web search adjusts `p_start`, weighted by `config.news.weight` (0 = pure stats, 1 = news dominates).
- **No PL history** (foreign signings, promoted-team squads): prior from price percentile within position — FPL's own pricing is a signal of expected role — plus `confidence: low` and an explicit flag. Never a silent guess.
- `E[minutes] = p_start × m_start + p_sub × m_sub`, with `m_start`/`m_sub` calibrated in the backtest (initial 80 / 20).

### 6.4 Team strength and fixture adjustment
FPL's granular strength fields are zeroed, so ratings are **derived**: attack and defence ratings from last season's goals for/against per game, normalized to the league mean, with separate home and away factors. Per-fixture FDR is used as a cross-check prior, not the primary signal. Promoted teams have no PL season to derive from → league-relegation-adjusted prior plus a low-confidence flag.

Expected goals conceded for a fixture drives clean-sheet probability via Poisson:

```
xGC_fixture = team_def_rating × opp_att_rating × home_away_factor
p_CS = exp(−xGC_fixture)
```

Attacking components scale by the opponent's defensive weakness over the same ratings.

### 6.4b Optional odds-based prior for promoted teams (`odds_provider`)
Price percentiles are a weak proxy for team strength, and market-implied totals are known to outperform them for team-level xG/xGC. `model/strength.py` therefore accepts an **optional** `odds_provider`, default `None`:

- **Method:** recover per-team implied goals by fitting a Poisson model to 1X2 plus Over/Under 2.5 lines.
- **Applies to:** promoted teams (no PL season to derive ratings from) and, indirectly, their players.
- **Default when unconfigured:** the price-percentile prior with `confidence: low` and an explicit flag. Deliberately optional — it affects ~3 teams, carries third-party ToS and availability risk, and must degrade gracefully in headless cron runs.
- **Per `fpl-data-fetch`:** odds are supplementary signal. Cite the source; never fabricate or estimate a missing line.

### 6.5 Double and blank gameweeks
Count fixtures per `(team, event)` from the full 380-fixture list. 0 → blank, xP contribution 0. 2+ → double, xP is the sum over both fixtures. Detection is generic, not hardcoded to a schedule. Cross-checked against MCP `get_blank_gameweeks` / `get_double_gameweeks`.

### 6.6 Defensive Contribution — verified against live data (2026-07-25)
`defensive_contribution` in `history_past` is a **raw qualifying-action count**, computed position-specifically. Confirmed by exact arithmetic:

- Rice (MID): CBI 127 + tackles 69 + recoveries 180 = **376** = `defensive_contribution`
- Saka (MID): 28 + 40 + 116 = **184** = `defensive_contribution`
- Lewis-Skelly: 17 + 8 + 30 = 55 but `defensive_contribution` = **25** = CBI + tackles only — he was a **DEF** in 2025/26 (reclassified MID for 2026/27), confirming the DEF formula excludes recoveries

That DC pays **direct** points was tested by residual analysis — actual points minus everything the standard categories explain:

| Player | Pos | Actual | Explained | Residual | DC/game |
|---|---|---|---|---|---|
| Rice | MID | 184 | 153 | +31 | 11.1 |
| Saka | MID | 157 | 143 | +14 | 7.4 |
| Eze | MID | 113 | 104 | +9 | 5.7 |
| Ødegaard | MID | 74 | 67 | +7 | 6.8 |

All residuals positive, correlating **0.774** with DC per game; Rice's +31 ≈ 15 matches × 2 pts for a DM averaging 11.1 against a 12 threshold. **Caveat:** appearances were approximated as `minutes/90`, which undercounts sub appearances and inflates residuals — the correlation is the reliable signal, not the absolute values. Corroborating structural evidence: FPL lists `defensive_contribution` in `element_stats` as its own labeled stat *alongside* `bps`, not within it.

**Implementation note:** the DC threshold is per-match, so it must be modeled as `p(actions ≥ threshold)` per fixture from a per-90 action rate — not by comparing a season total against the threshold.

## 7. Backtest — the trust gate

Two tiers, because neither alone is sufficient.

**Tier 1 — API aggregates (`backtest/aggregate.py`).** Multi-season walk-forward over `history_past` (up to 8 seasons/player): predict season N per-90 and minutes from seasons < N. Validates the baseline, shrinkage constant `k`, and minutes model. No external dependency. Doubles as an integrity check on the archive — if the archive's per-GW rows don't sum to `history_past` season totals, the archive is untrustworthy and Tier 2 is void.

**Tier 2 — per-GW (`backtest/gw_level.py`).** Train on 2024-25, test on 2025-26 using per-GW rows (`xP`, xG, xA, minutes, bps, fixture). Validates what Tier 1 structurally cannot: fixture-difficulty adjustment, form-decay `h` and `w_max`, and captaincy.

**Metrics:** MAE and RMSE on points; Spearman rank correlation **per position** (rank quality matters more than absolute error for selection); calibration curve; top-20 overlap between predicted and actual; captaincy hit rate.

**Baselines it must beat, or it isn't trusted:**
1. Naive "last season points-per-90" with no adjustments
2. FPL's own `xP` column present in the per-GW data

**Gate:** if the model does not beat both baselines on per-position rank correlation, the weekly report states the model is low-confidence and names the components that failed validation. The backtest result is reported to the user before GW1 — it is not a silent internal check.

## 8. Optimization

Per the `squad-optimizer` skill (authoritative), using `pulp` MILP.

### Mode 1 — full build
Maximize xP over a 5-GW horizon subject to: budget ≤ £100.0m; exactly 15 players as 2 GK / 5 DEF / 5 MID / 3 FWD; **max 3 per real club**; a valid XI of 1 GK + 3–5 DEF + 2–5 MID + 1–3 FWD = 11; captain = highest-xP starter (2×), vice = second.

**Documented refinement to the skill.** The skill reads as two stages — maximize squad xP, then pick the XI. Implemented literally, that overspends budget on bench players who score nothing. Instead this is a **single joint MILP** maximizing `XI xP + bench_weight · bench xP`, with `bench_weight` decaying per bench slot (default `[0.15, 0.10, 0.05, 0.02]`). This is faithful to the skill's intent — maximize points actually scored — while directing budget to the XI. Flagged here because the skill is authoritative and this is a deliberate interpretation, not an accident.

### Mode 2 — weekly transfers
Given the current 15 (from `entry/{id}/event/{gw}/picks/`) and the available free transfers: maximize `net xP gain over horizon − 4 × paid transfers`. **Always report the 0-transfer baseline** so the gain from moving is visible. Same budget/formation/club constraints on the post-transfer squad. Dead code until a team ID exists — must not block Mode 1.

**Free transfer rules (current FPL mechanics).** FTs **accumulate up to a cap of 5** (not the historical 2), and are **not reset by playing a Wildcard or Free Hit**. Consequences for the design:

- `0 ≤ FT ≤ 5` as a bound on the decision variables.
- **Search depth is `FT_available + max_paid_hits`, not a fixed 3.** A fixed cap of 3 would discard *free* moves whenever more than 3 FTs are banked — leaving xP on the table at zero cost. This was a bug in the first draft of this spec.
- Chip weeks **preserve** the FT balance and still accrue the weekly +1, up to the cap.

**FT count is not available from any public endpoint.** `my-team/` exposes it but is auth-gated and off-limits (§12). So FT lives in persisted local state (`fpl/state.py` → `data/state.json`), advanced as:

```
FT_next = min(5, FT_current + 1 − transfers_made_this_gw)
```

Seeded once by the user, and reconcilable against `entry/{id}/history/` → `event_transfers` per event as a drift check. If state is missing or fails reconciliation, the report must state the FT count is assumed rather than known, and show the transfer recommendation at both the assumed and the conservative FT count.

### Chip advisor
Heuristic flags with the tradeoff stated, never an unexplained auto-recommendation: **Wildcard** on a fixture swing or multiple simultaneous squad problems; **Bench Boost** only when all 15 have good fixtures and no bench xP is near-zero; **Triple Captain** for a standout with a DGW or a very soft single fixture; **Free Hit** for a blank GW or a one-off double that doesn't justify a wildcard.

## 9. Report

Exactly the `weekly-report` skill's structure: header with GW and deadline, Starting XI by position with formation, ordered bench, captain/vice with one line of reasoning, transfers with net xP after any hit, chip watch, budget, and flags.

Rules enforced in code where possible: one sentence of "why" per major decision; uncertainty surfaced explicitly rather than presented as settled; **recommendation phrasing only** — "recommend"/"suggest", never "transferred"/"captained", because the user applies it manually. Scannable in under a minute.

Every `flags` entry from the xP frame for a selected player must appear in the report's Flags section, with its source cited when web-sourced.

## 10. Config

`config.yaml`:

```yaml
budget: 100.0
horizon_gw: 5
risk:
  profile: balanced          # balanced | template | differential
  ownership_weight: 0.0      # >0 favours template, <0 favours differentials
news:
  weight: 0.5                # 0 = pure stats, 1 = news dominates minutes prior
  max_age_hours: 48          # older team news is ignored
model:
  form_half_life_gw: 3       # fitted by backtest
  form_max_weight: 0.6       # fitted by backtest
  shrinkage_minutes: 900     # fitted by backtest
optimizer:
  max_paid_hits: 2           # search depth = FT_available + this (FT caps at 5)
  hit_cost: 4
  bench_weight: [0.15, 0.10, 0.05, 0.02]
odds:
  provider: null             # null = price-percentile fallback for promoted teams
data:
  cache_ttl_hours: 6
  cache_ttl_matchday_hours: 1
entry_id: null               # set after the first squad is saved
free_transfers: 1            # local state seed; FT caps at 5, survives chips
```

Validated on load with clear errors; every value above has a working default.

## 11. Headless operation

`run_gameweek.py` imports no MCP client and loads no skill — it is the cached data plus the HTTP client. The cron path therefore works **by construction**, not by verification. `claude -p "run the weekly gameweek update"` will still be tested end-to-end as requested, but nothing depends on that test passing.

This matters because the MCP server is user-scoped rather than project-scoped (§2), so its availability in a headless invocation is an assumption worth not relying on.

CLI: `--mode {1,2}`, `--gw N`, `--no-refresh` (cache only), `--deep` (full per-player sweep), `--backtest`.

## 12. Guardrails

1. No endpoint or tool requiring a login/session cookie. `my-team/` never called. `get_my_team` and `check_fpl_authentication` are off-limits by default, not "used carefully".
2. No writes, POSTs, transfers, chip activations, or lineup changes. Any write-capable tool discovered later is off-limits until explicitly cleared.
3. Never fabricate a number. Missing data is reported as missing.
4. Uncertainty is stated, not smoothed over — "limited data on this summer signing", "start uncertain per team news".

## 13. Known risks and open uncertainties

| Risk | Handling |
|---|---|
| **Form model unvalidated at GW1** — `w_form = 0`, so the entire form mechanism is untested in the first live use | Tier 2 backtest fits it on historical GW data; report states the GW1 squad rests on season baselines alone |
| **Promoted teams and foreign signings have no PL data** | Price-percentile prior + `confidence: low` + explicit flag. Expect these to be the model's weakest picks |
| **Derived team strength is a proxy** for the zeroed FPL fields | Cross-checked against per-fixture FDR; disagreement is surfaced |
| **Four weeks of drift** before the deadline — prices, ownership, and especially team news will move | Any squad built now is a structural dry run. The real build runs in deadline week |
| **Archive is a third-party dependency** | Confined to `backtest/`, integrity-checked against `history_past` totals, and irrelevant from GW2 onward |
| **Pre-season news vacuum** — `news` empty, `chance_of_playing` null for all 558 | GW1 minutes model leans on last season's `starts` plus web-sourced team news close to the deadline |
| **FT balance is unobservable** from public endpoints — the only source is auth-gated and off-limits | Tracked in `data/state.json`, reconciled against `entry/{id}/history/` `event_transfers`. When unknown, the report says so and shows both assumed and conservative FT counts rather than presenting a guess as fact |
| **Odds provider is third-party** if enabled — ToS and availability risk | Optional and default-off; unconfigured falls back to price percentile. Never a hard dependency of the weekly or headless run |

## 14. Build order

1. **Data layer** — client, cache, normalize, store. Tests on freshness logic, price conversion, ID resolution, stale-fallback.
2. **Model** — strength → minutes → scoring → fixtures → bps → xp. Tests on the frame contract and DGW/BGW arithmetic.
3. **Backtest** — Tier 1 then Tier 2. **Gate: report accuracy to the user before trusting the model.**
4. **Optimizer** — Mode 1 first, then lineup/captain, chips, then Mode 2 with `state.py` for the FT balance. Tests asserting constraints that must never break: budget ceiling, 2/5/5/3, max-3-per-club, valid formation, and a known-optimum on a small synthetic pool. Mode 2 tests must cover the FT state machine at the 5 cap and across a chip week.
5. **Report + `run_gameweek.py`** — then verify the headless path.

Testing is TDD throughout. The constraint tests in step 4 are the highest-value tests in the codebase: an invalid squad is worse than no recommendation, because it can't be applied at all.
