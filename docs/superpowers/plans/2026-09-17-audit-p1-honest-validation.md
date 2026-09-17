# P1 — Honest Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the evidence this tool is tuned on trustworthy — so that a measured improvement is a real one, and no forecast is scored against data it was allowed to see.

**Architecture:** An append-only forecast manifest makes the ledger immutable and says which forecast was actually *acted on*; calibration and scoring read only that one. A point-in-time snapshot store records what was known at each deadline, going forward. A new sequential replay engine walks one manager's state through a season — squad, bank, purchase prices, free transfers, hits, chips and Free Hit restoration — so competing policies can be compared on the decisions they would really have made, with the free-rebuild oracle kept as a clearly labelled ceiling.

**Tech Stack:** Python 3.13, pandas, PuLP, pytest. No new dependencies.

**Spec:** `docs/fpl-model-optimizer-audit-2026-09-17.md` — items B13, B14, B15, B16, and the P1 acceptance gate.

## Global Constraints

- No new third-party dependencies.
- `python -m pytest -q` passes at the end of every task. Baseline entering P1: **576 passed**.
- The weekly pipeline must never import `fpl/data/archive.py`.
- A replayed forecast is never allowed to become the scored record of a live gameweek. `origin` is load-bearing, not decoration.
- Commit after every task, audit item id in the body, with the session's attribution line.

## Honest limitation, stated once and carried in the code

Point-in-time `bootstrap-static` for GW1–GW4 of 2026/27 **does not exist** — the cache keeps three snapshots per slug and those gameweeks are long past. So B13 splits in two:

1. the **machinery** (snapshot capture + a replay engine that consumes it) is built now and is fully testable;
2. a replay of *this* season's past gameweeks still reads today's bootstrap for price/status/news, and must therefore keep saying so.

Task 4 builds the capture so the limitation expires; Task 5 builds the engine and makes the contamination a typed, reported field rather than a docstring.

## File Structure

**New files**

| File | Responsibility |
|---|---|
| `fpl/backtest/manifest.py` | Append-only record of every forecast version: origin, deadline, actioned marker. Selection rules live here. |
| `fpl/data/snapshots.py` | Point-in-time capture of the payloads a run read, keyed by gameweek, so a later replay can reconstruct the deadline's inputs. |
| `fpl/backtest/replay.py` | Sequential manager-state replay: state, policies, and the season walk. |
| `tests/test_manifest.py`, `tests/test_snapshots.py`, `tests/test_replay.py` | Their tests. |

**Modified**

| File | Change |
|---|---|
| `fpl/backtest/ledger.py` | `save_predictions` writes a manifest entry; `load_predictions` selects the actioned/pre-deadline non-replay version; `MODEL_VERSION` from git. |
| `fpl/model/calibration.py` | `scored_history` selects through the manifest. |
| `fpl/pipeline.py` | Passes the deadline and origin into `save_predictions`; coverage gate before optimization. |
| `run_gameweek.py` | `--confirm` marks the acted-on forecast. |
| `fpl/data/client.py` | `element_summaries` reports per-player fetch status. |
| `fpl/data/normalize.py` | Distinguishes `new_to_league` / `no_history` / `fetch_failed`. |
| `fpl/backtest/gw_level.py` | Captaincy measured on an owned XI; `trust_gate` refuses a rate-only diagnostic. |
| `scripts/run_backtest.py` | Tier 2 keeps DNPs and uses expected minutes; renamed as a component diagnostic. |
| `scripts/run_walkforward.py` | Reports the oracle ceiling and the sequential replay side by side. |

---

## Task 1: Append-only forecast manifest

**Audit item:** B14 (High)

**Interfaces produced** (`fpl/backtest/manifest.py`):
- `MANIFEST_FILE = "manifest.jsonl"`
- `record_version(root, *, gw, version, created_at, origin, model_version, config_hash, deadline=None) -> dict` — appends one line, never rewrites.
- `entries(root, gw=None) -> list[dict]` — every recorded version, oldest first.
- `mark_actioned(root, gw, version=None, when=None) -> dict | None` — appends an `actioned` record for that version (the newest if unnamed).
- `select_version(root, gw, deadline=None) -> dict | None` — the confirmed actioned version; else the newest `origin="live"` version strictly before `deadline`; else the newest live version. **Never** returns `origin="replay"`.
- `model_version() -> str` — `git describe`-free: short SHA plus `-dirty`, falling back to the hand-set constant outside a repo.

Key decisions to encode in docstrings: the manifest is append-only because the whole point is that a later run cannot rewrite what an earlier one recorded; and `select_version` excludes replays because a replay is built from data the live model never had.

- [ ] **Step 1: Write `tests/test_manifest.py`** covering: an appended version is readable back; a second write for the same gameweek does not replace the first; `select_version` prefers the actioned version over a newer one; with no actioned marker it takes the newest live version strictly before the deadline; a replay version is never selected even when it is the only one; `model_version()` returns a non-empty string.
- [ ] **Step 2: Run it, confirm `ModuleNotFoundError`.**
- [ ] **Step 3: Implement `fpl/backtest/manifest.py`.**
- [ ] **Step 4: Tests pass; full suite passes.**
- [ ] **Step 5: Commit** — `feat: append-only forecast manifest with origin and actioned marker`.

---

## Task 2: Scoring and calibration read only the acted-on forecast

**Audit item:** B14 (High, second half)

**Interfaces:**
- `ledger.save_predictions(...)` gains `origin: str = "live"` and `deadline: str | None = None`; it records a manifest entry for every write and returns the version path.
- `ledger.load_predictions(gw, root, deadline=None)` resolves through `manifest.select_version` and falls back to `gw{n}.parquet` only when no manifest exists (files written before this task).
- `run_gameweek.py --confirm` calls `manifest.mark_actioned(root, gw)`.
- `scripts/run_walkforward.py` passes `origin="replay"`.

- [ ] **Step 1: Write the tests.** A replay write after a live write must not change what `load_predictions` returns. A post-deadline live re-run must not displace the actioned forecast. `scored_history` must skip a gameweek whose only forecast is a replay.
- [ ] **Step 2: Confirm they fail.**
- [ ] **Step 3: Implement.** Keep `gw{n}.parquet` as the convenience pointer; make selection authoritative.
- [ ] **Step 4: Full suite.**
- [ ] **Step 5: Commit** — `fix: score and calibrate on the forecast that was acted on`.

---

## Task 3: Tier 2 stops handing the model future minutes

**Audit item:** B15 (High validation risk)

**What is wrong:** `scripts/run_backtest.py` filters to `minutes > 0` and multiplies predictions by the minutes that actually occurred. That gives the rate model future playing time and deletes every nonappearance — the dominant source of FPL error — while the result feeds `trust_gate`.

**Interfaces:**
- `gw_level.captaincy_hit_rate(pred_by_gw, actual_by_gw, owned_by_gw=None)` — when `owned_by_gw` is given, the captain is the highest projection among the players **owned** that week, and a hit means he was that XI's top actual scorer. The old global-argmax behaviour stays only when `owned_by_gw` is None, documented as a pool diagnostic.
- `gw_level.trust_gate(model, naive, fpl_xp, *, full_pipeline: bool)` — refuses to return `trusted=True` when `full_pipeline` is False, with a summary naming the reason.
- `scripts/run_backtest.py` — Tier 2 retains all registered player-gameweek rows and scores full xP from the point-in-time minutes model; the rate-only variant is printed separately and labelled `component diagnostic (rate model only)`.

- [ ] **Step 1: Tests** — a DNP row survives into the Tier 2 frame; `captaincy_hit_rate` with `owned_by_gw` picks from the owned set, not the pool; `trust_gate(..., full_pipeline=False)` is never trusted.
- [ ] **Step 2: Confirm failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite.**
- [ ] **Step 5: Commit** — `fix: Tier 2 keeps nonappearances and cannot gate production trust`.

---

## Task 4: A failed fetch is not a newcomer

**Audit item:** B16 (High during an outage)

**Interfaces:**
- `client.element_summaries(...)` additionally populates `self.fetch_failures: set[int]`; the returned dict is unchanged, so no caller breaks.
- New `normalize.history_status(players, summaries, fetch_failed) -> pd.DataFrame` with one row per player and a `history_status` in `{"established", "new_to_league", "no_history", "fetch_failed"}`.
- New `pipeline.coverage_gate(players, summaries, fetch_failed, owned_ids, min_coverage=0.99) -> list[str]` returning blocking reasons; `run` raises `DataCoverageError` when an **owned** player's summary failed, or overall coverage falls below the floor.
- `DataCoverageError` lives in `fpl/data/client.py` so both layers can import it without a cycle.

Deliberate scope note: `fetch_failed` players keep the newest cached row when one exists — that is strictly better than zeroing them — and only a player with no cache at all reaches the gate.

- [ ] **Step 1: Tests** — a failed fetch is reported distinctly from a genuine newcomer; a failed fetch for an owned player aborts the run; a failed fetch for an irrelevant player below the coverage floor aborts; above it, warns.
- [ ] **Step 2: Confirm failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite.**
- [ ] **Step 5: Commit** — `fix: distinguish a failed fetch from a genuine newcomer and gate on coverage`.

---

## Task 5: Point-in-time snapshots

**Audit item:** B13 (Critical validation risk, first half)

**Interfaces** (`fpl/data/snapshots.py`):
- `capture(root, gw, *, bootstrap, fixtures, deadline, sources) -> Path` — writes `data/snapshots/gw{n}/` with `bootstrap.json`, `fixtures.json` and `meta.json` (`captured_at`, `deadline`, `gw`, `sources`, `final_through`). Idempotent per gameweek: the **first** capture before a deadline wins, because that is the one that describes the deadline.
- `load(root, gw) -> dict | None` — `{"bootstrap":…, "fixtures":…, "meta":…}`.
- `available(root) -> list[int]`.
- `is_point_in_time(root, gw) -> bool` — True only when a snapshot exists AND its `captured_at` precedes its `deadline`.

`pipeline.run` calls `capture` once per run, before optimization. Element summaries are **not** copied — they are already per-round and the cache keeps them — but their source stamps are recorded in `meta.json`.

- [ ] **Step 1: Tests** — a capture round-trips; a second capture for the same gameweek does not overwrite the first; `is_point_in_time` is False for a capture taken after the deadline.
- [ ] **Step 2: Confirm failures.**
- [ ] **Step 3: Implement, and wire `pipeline.run`.**
- [ ] **Step 4: Full suite.**
- [ ] **Step 5: Commit** — `feat: capture a point-in-time snapshot each run`.

---

## Task 6: Sequential manager-state replay

**Audit item:** B13 (Critical validation risk, second half)

**Interfaces** (`fpl/backtest/replay.py`):

```python
@dataclass
class ManagerState:
    squad: list[int]
    bank: float
    purchase_prices: dict[int, float]
    free_transfers: int = 1
    chip_events: list[dict] = field(default_factory=list)
    base_squad: list[int] = field(default_factory=list)   # Free Hit
    base_bank: float = 0.0
    base_purchase_prices: dict[int, float] = field(default_factory=dict)
    freehit_event: int | None = None
    points: float = 0.0
    hits: int = 0

@dataclass
class GameweekResult:
    gw: int
    points: float          # after autosubs, captain and hits
    gross_points: float
    hit_cost: int
    transfers: int
    chip: str | None
    squad: list[int]
    xi: list[int]
    captain: int
    free_transfers_after: int
    bank_after: float
```

- `Policy = Callable[[pd.DataFrame, ManagerState, int, Config], Decision]` where
  `Decision = (squad_ids, starting_ids, chip)`.
- Built-in policies: `hold_policy`, `expected_points_policy`, `rank_policy`, `oracle_rebuild_policy`.
- `replay_season(xp_by_gw, actuals, state, policy, cfg, *, contaminated: bool) -> tuple[list[GameweekResult], ManagerState]`.
- `compare_policies(...) -> pd.DataFrame` — one row per policy with total points, hits paid, chips used, and the mean weekly edge against a supplied field average.

Rules the engine must honour, each with its own test:
1. Transfers are charged against `state.free_transfers` and the balance advances through `fpl.state.ft_after_moves` — the same function the live tool uses, so a bug cannot exist in one and not the other.
2. A Wildcard or Free Hit makes the week's transfers free.
3. A Free Hit's squad scores that gameweek and is then **discarded**: the next gameweek starts from `base_squad`/`base_bank`/`base_purchase_prices`.
4. Selling prices use `fpl.optimize.transfers.selling_price` against recorded purchase prices.
5. Chip legality goes through `fpl.chips.chip_available` against the accumulated `chip_events`.
6. Scoring goes through the existing `walkforward.realised_score`, so autosubs and the armband are applied exactly as the live scorer applies them.
7. `oracle_rebuild_policy` is labelled in the comparison output as a ceiling, never as a policy anyone could execute.

- [ ] **Step 1: Tests** (`tests/test_replay.py`) — one per numbered rule above, plus: a hit is charged once and only once; `compare_policies` marks the oracle row; a contaminated replay says so in its output.
- [ ] **Step 2: Confirm failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite.**
- [ ] **Step 5: Commit** — `feat: sequential manager-state replay for policy comparison`.

---

## Task 7: Wire the replay into the walk-forward script and gate the claim

**Audit item:** B13 acceptance gate

- `scripts/run_walkforward.py` gains `--policies expected,rank,hold` and prints the sequential comparison **above** the scratch-rebuild number, which is relabelled `ORACLE CEILING (not executable)`.
- The script refuses to describe any replay as evidence unless `snapshots.is_point_in_time` holds for every replayed gameweek; otherwise it prints the contamination banner and calls the result an upper bound.

- [ ] **Step 1:** Update the script and its docstring.
- [ ] **Step 2:** Run `python scripts/run_walkforward.py --through 4 --no-save` and check the banner appears (it must, for this season).
- [ ] **Step 3:** Full suite.
- [ ] **Step 4: Commit** — `feat: report executable policies above the oracle ceiling`.

## Self-Review

Spec coverage: B13 → Tasks 5, 6, 7. B14 → Tasks 1, 2. B15 → Task 3. B16 → Task 4.
Ordering: 1 → 2; 5 → 6 → 7; 3 and 4 are independent.
Type consistency: `origin` is a string in both `save_predictions` and `manifest.record_version`; `select_version` returns the same dict shape `entries` yields.
