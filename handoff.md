# Handoff — FPL audit remediation

**Status:** P0 complete (576 tests). **P1 in progress** — 5 of 7 tasks done, suite at **609 passed**.
**Last updated:** 2026-09-17
**Branch:** `master` — 14 commits, `370de0e..22ff9ba`

## What this work is

`docs/fpl-model-optimizer-audit-2026-09-17.md` is a full audit of the model and
optimizer: 17 bugs (B1–B17), 11 robustness gaps (R1–R11), and a recommended
implementation order P0 → P3.

Scope agreed for **this** pass, and delivered in full:

- **P0** — B1, B2, B3, B4, B5 ("make actions legal and state reversible")
- **The self-contained correctness bugs** — B9, B10, B11, B12, B17, R11

For B4 the decision was explicitly to build the **real** chip squads (Wildcard
returns its rebuild, Free Hit returns an optimized one-week squad), rather than
the audit's cheaper interim option of demoting chip advice to a timing flag.

The plan that was executed: `docs/superpowers/plans/2026-09-17-audit-p0-and-correctness.md`.
It carries the reasoning that is not in the diffs — read it before changing any
of this.

## Architecture introduced

Chip handling is split three ways, and the split is load-bearing:

| Module | Owns |
|---|---|
| `fpl/chips.py` (new) | **Legality** — which chip may be played in which gameweek, read only from the dated `State.chip_events` record. No pandas, no optimizer imports. |
| `fpl/optimize/chips.py` | **Timing** — whether a *legal* chip is worth playing now. |
| `fpl/optimize/actions.py` (new) | **Effect** — the Wildcard rebuild, the one-week Free Hit squad, Bench Boost and Triple Captain marginal values. |
| `fpl/state.py` | **Record** — what was played and when, plus the permanent squad/bank/purchase prices a Free Hit must not destroy. |

## What changed, commit by commit

| Commit | Audit | Change |
|---|---|---|
| `f5c...` chip legality | B1 | New `fpl/chips.py`: two chip windows (GW1–19, GW20–38), one use per chip per half, first set expires, WC/FH banned in an entry's opening gameweek, one active chip per gameweek, no back-to-back Free Hits. |
| FT carry | B2 | `ft_after_moves` returns the **same** balance after WC/FH, not balance + 1. FPL consumes the gameweek's own free transfer to activate the chip. |
| Free Hit state | B3 | `State` gains `base_squad` / `base_bank` / `base_purchase_prices` / `freehit_event`. `record_transfers` preserves them on a FH confirmation; `resolve_current_squad` restores from them on the next planning run. |
| `dfcd08a` | B1 | `advise_chips` takes dated `chip_events` + `first_event`, not a name list, and names the rule that blocked a chip. |
| `8dcb86d` | B4 | New `fpl/optimize/actions.py` — each chip as a complete action. |
| `51956a4` | B4 | Pipeline replaces squad/XI/transfers/lineup with the chip's own when one is advised; report says when the 15 shown are temporary. `Recommendation.chip_squad` / `.chip_temporary`. |
| `7a33dcc` | B4 | `--confirm` refuses (exit 1) without `--applied-chip` whenever the run advised a chip. `main()` now takes `argv` so it is testable. |
| `f8057d7` | B5, B17 | Transfer enumeration reserves a slot for **every** transfer count before spending the rest on alternatives. Hold plan may keep an existing four-from-one-club overage; any transfer plan may not. |
| `8a56af5` | B9 | Availability caps `p_start`, `p_play`, `p_60` and `e_minutes` together. Unavailable → all zero. Doubtful → the cameo branch is scaled too. |
| `b387fdd` | B10 | Calibration applies one intercept per gameweek **with a fixture**, and rebuilds `xp_next1` / `xp_next5` / `xp_horizon` from the calibrated weeks. Blanks stay blank. |
| `6f93808` | B11 | `history_current_frame` gains `matches_played` (fixture rows) beside `gws_played` (distinct rounds); the start denominator uses matches. |
| `5bb5cb9` | R11 | `bench_floor_xp` defaults to `0.0` and is now explicit in `config.yaml`. |
| `2115439` | B12 | Tier 1 backtest replays by season cutoff; prior and naive baseline come only from pre-cutoff seasons; adds `by_cutoff`. |
| `22ff9ba` | — | `tests/test_chip_acceptance.py`, the P0 acceptance gate. |

## Verification

- `python -m pytest -q` → **576 passed** (baseline 510; 66 added, several rewritten).
- `python run_gameweek.py --mode 2 --gw 5 --no-refresh` renders a full report and
  leaves `data/state.json` untouched (planning run).

## Behaviour changes to be aware of before the next live run

1. **`bench_floor_xp` is now 0.0** (`config.yaml`). The GW5 smoke run benched
   D.Essugo at 0.0 xP, which the old 2.5 floor would have forbidden. If a Bench
   Boost is still being prepared, set it back to `2.5` in `config.yaml` — the
   point of the change is that the setting is now visible and deliberate, not
   that 0.0 is right for every week.
2. **`--confirm` now requires `--applied-chip`** whenever the run advised a chip.
   Any cron or script that confirms unattended must pass `--applied-chip none`.
3. **Free-transfer balances after a chip week are one lower** than the tool
   previously reported. That is the correction, not a regression.
4. **`pipeline.run`'s `chips_used=` parameter is gone**, replaced by
   `chip_events=` (dated dicts) and `first_event=`.

## Open questions

1. **`FREE_HIT_MIN_GAP`** in `fpl/chips.py` encodes the audit's claim that two
   Free Hits cannot be played in consecutive gameweeks. That is the one rule
   taken on the audit's authority rather than verified independently. It is
   isolated in one named constant — set it to `0` to disable.
2. **Undated `chip_events`** (state files written before `chip_events` existed)
   are charged to the **first** chip window. That can only forfeit a chip
   already spent; the alternative would forfeit one still held.
3. **`data/state.json` has no `base_*` fields yet.** They are written the next
   time a Free Hit is confirmed. Until then `freehit_event` is `None` and the
   restoration branch is inert, which is correct.

## P1 progress (plan: `docs/superpowers/plans/2026-09-17-audit-p1-honest-validation.md`)

| # | Task | Audit | Status |
|---|---|---|---|
| 1 | Append-only forecast manifest (`fpl/backtest/manifest.py`) | B14 | **done** |
| 2 | Ledger serves the acted-on forecast; replays never served | B14 | **done** |
| 3 | Tier 2 keeps DNPs, uses training-season minutes, cannot gate trust | B15 | **done** |
| 4 | Failed fetch ≠ newcomer; `coverage_gate` aborts the run | B16 | **done** |
| 5 | Point-in-time snapshots (`fpl/data/snapshots.py`) | B13 | **done** |
| 6 | Sequential manager-state replay (`fpl/backtest/replay.py`) | B13 | **IN PROGRESS** — module + `tests/test_replay.py` written, tests not yet run |
| 7 | Wire replay into `scripts/run_walkforward.py` above the oracle ceiling | B13 | not started |

### P1 design notes worth keeping

- **`fpl/backtest/manifest.py`** is append-only JSONL. `select_version` order of
  authority: explicitly actioned → newest live version strictly before the
  deadline → newest live. A `origin="replay"` version is **never** selected.
  `--confirm` calls `mark_actioned`. `MODEL_VERSION` is now the git short SHA
  plus `-dirty`, not a hand-set date.
- Ledger version ids are sub-second (`%Y%m%dT%H%M%S%fZ`); second resolution
  collided when a planning run and a confirmation ran in the same second.
- **`trust_gate(..., full_pipeline=False)` is the default** and always returns
  `trusted=False`. Existing callers had to be updated to assert `True`.
- **`coverage_gate`** raises `DataCoverageError` (defined in `fpl/data/client.py`)
  when an *owned* player's summary is unreadable, or pool coverage < 99%.
- **Snapshots**: first capture per gameweek wins. `is_point_in_time` is False
  unless `captured_at < deadline`. GW1–4 of this season can never be
  point-in-time — that data is gone — so `contamination_note()` exists to say so.
- **`fpl/backtest/replay.py`** deliberately reaches the live rules through the
  live functions (`ft_after_moves`, `selling_price`, `chip_available`,
  `realised_score`) so a rule cannot be right in the replay and wrong in
  production. `oracle_rebuild_policy` is the old free-weekly-rebuild behaviour,
  kept but marked `executable=False`.

## Still open from the audit — each needs its own plan

Nothing below was touched.

**P1 — honest validation (the audit calls this the prerequisite for tuning):**
- B14 immutable actioned forecasts; exclude replays from calibration
- B13 point-in-time snapshots and a sequential manager-state replay
- B15 Tier 2 uses actual minutes and excludes DNPs while gating production trust
- B16 a partial element-summary fetch is indistinguishable from a newcomer

**P2 — objective and simulation alignment:**
- B6 rank reranking discards the multi-gameweek transfer objective
- B7 optimizer, simulator and reported captain are three different decisions
- B8 one fixed XI and bench order across the whole horizon
- R1 one-week median-beat target; R2 the rival field; R3 incoherent match outcomes

**P3 — information, then sophistication:**
- R4 confidence never affects the distribution
- R5 event-specific, team-coherent minutes
- R6 cold starts and stale overrides
- R7 attacking fixture adjustment double-counts team quality
- R8 no multi-period transfer model or FT option value
- R9 chip timing beyond the horizon uses structure, not value
- R10 goalkeeper/bonus bias is structural

Suggested next step if continuing: **P1 first**. The audit's argument is that a
trustworthy backtest is worth more than another forecasting feature, because it
is the only thing that says whether a feature should influence a transfer at all.
