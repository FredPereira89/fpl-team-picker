# Handoff — FPL audit remediation

**Status:** in progress — P0 complete (tasks 1-9 of 15), suite at 542 passed
**Last updated:** 2026-09-17
**Branch:** `master`

## What this work is

`docs/fpl-model-optimizer-audit-2026-09-17.md` is a full audit of the model and
optimizer: 17 bugs (B1–B17), 11 robustness gaps (R1–R11), and a recommended
implementation order P0 → P3.

Agreed scope for **this** pass:

- **P0 in full** — B1, B2, B3, B4, B5 ("make actions legal and state reversible")
- **Plus the self-contained correctness bugs** — B9, B10, B11, B12, B17, R11

Explicitly **out of scope** for this pass (each needs its own plan):
B6, B7, B8, B13, B14, B15, B16, R1–R10 — i.e. the whole of P1 (honest
validation: point-in-time snapshots, sequential manager-state replay, immutable
actioned forecasts), P2 (objective alignment: stop one-week rank reranking,
per-event lineup variables, coherent match scenarios) and P3 (event-specific
minutes, team-goal/player-share attack model, rebuilt BPS, multi-period MILP).

A decision that was made explicitly: for B4, build the **real** chip squads
(Wildcard returns its rebuild, Free Hit returns an optimized one-week squad),
rather than the audit's cheaper interim option of demoting chip advice to a
non-actionable timing flag.

## The plan

`docs/superpowers/plans/2026-09-17-audit-p0-and-correctness.md` — 15 tasks, each
with its tests, its implementation, and its commit message. Read it before
touching anything; it carries the reasoning that is not in the diffs.

## Architecture introduced by this work

Chip handling is split three ways, and the split is load-bearing:

| Module | Owns |
|---|---|
| `fpl/chips.py` (new) | **Legality** — which chip may be played in which gameweek, read only from the dated `State.chip_events` record. No pandas, no optimizer imports. |
| `fpl/optimize/chips.py` | **Timing** — whether a *legal* chip is worth playing now. |
| `fpl/optimize/actions.py` (new) | **Effect** — what the chip actually does to the squad: the Wildcard rebuild, the one-week Free Hit squad, Bench Boost and Triple Captain marginal values. |
| `fpl/state.py` | **Record** — what was played and when, plus the permanent squad/bank/purchase prices a Free Hit must not destroy. |

## Task status

| # | Task | Audit | Status |
|---|---|---|---|
| 1 | Chip legality rules (`fpl/chips.py`) | B1 | **done** |
| 2 | WC/FH free-transfer carry | B2 | **done** |
| 3 | Free Hit preserves the permanent squad | B3 | **done** |
| 4 | Advisor reads dated chip events | B1 | **done** |
| 5 | Chip actions module | B4 | **done** |
| 6 | Pipeline returns the chip's real squad | B4 | **done** |
| 7 | `--confirm` requires an explicit chip | B4 | **done** |
| 8 | Stratified transfer-count enumeration | B5 | **done** |
| 9 | Club-cap real-transfer exception | B17 | **done** |
| 10 | Unavailable players score zero everywhere | B9 | not started |
| 11 | Per-event calibration | B10 | not started |
| 12 | Start rate counts matches, not gameweeks | B11 | not started |
| 13 | Bench floor off by default | R11 | not started |
| 14 | Tier 1 backtest season cutoffs | B12 | not started |
| 15 | P0 acceptance gate | — | not started |

Ordering constraints: 1 → 4 → 6, and 2 → 3 → 7, and 8 → 9 (same function).
Tasks 10–14 are independent of everything else and of each other.

## Test baseline

`python -m pytest -q` → **510 passed** before this work started. Every task ends
with the full suite green.

## Open questions for whoever continues

1. **`FREE_HIT_MIN_GAP`** in `fpl/chips.py` encodes the audit's claim that two
   Free Hits cannot be played in consecutive gameweeks. That is the one rule
   taken on the audit's authority rather than verified against the code or the
   FPL site. It is isolated in one named constant — set it to `0` to disable.
2. **Undated `chip_events`** (from state files written before `chip_events`
   existed) are charged to the **first** chip window. That can only forfeit a
   chip already spent; the alternative would forfeit one still held.
