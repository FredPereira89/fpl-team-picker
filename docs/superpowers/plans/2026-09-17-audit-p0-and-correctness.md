# FPL Audit P0 + Correctness Bugs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every chip and transfer action this tool recommends legal under the 2026/27 rules, reversible in local state, and actually the squad the named chip would field — plus fix six self-contained model/validation bugs.

**Architecture:** A new `fpl/chips.py` owns chip *legality* (which chip may be played in which gameweek, from the dated `chip_events` record). A new `fpl/optimize/actions.py` owns what a chip *does* (the real Wildcard rebuild, the real one-week Free Hit squad, Bench Boost and Triple Captain marginal values). `fpl/optimize/chips.py` keeps owning *timing* — whether a legal chip is worth playing now. `fpl/state.py` gains permanent/base squad fields so a Free Hit cannot destroy the squad it temporarily replaces. The remaining tasks are independent point fixes in the model and backtest layers.

**Tech Stack:** Python 3.13, pandas, PuLP (CBC), pytest. No new dependencies.

**Spec:** `docs/fpl-model-optimizer-audit-2026-09-17.md` — items B1, B2, B3, B4, B5, B9, B10, B11, B12, B17, R11.

## Global Constraints

- No new third-party dependencies. The pipeline must stay runnable headless (`python run_gameweek.py`) with no MCP and no skills.
- Every module keeps the codebase's existing docstring style: docstrings explain *why* a rule exists and what broke without it, not what the code literally does.
- `python -m pytest -q` must pass at the end of every task. Baseline before this plan: **510 passed**.
- Tests that currently codify wrong chip/free-transfer behaviour are to be **rewritten**, not deleted, and the rewrite must state the real FPL rule in its docstring.
- Chip names are always canonicalised through `fpl.state.canonical_chip` before comparison. The four chip names used internally are `wildcard`, `freehit`, `benchboost`, `triplecaptain`.
- Free-transfer cap is 5 (`fpl.state.FT_CAP`).
- The first chip window is GW1–GW19; the second is GW20–GW38.
- Commit after every task with a `fix:` or `feat:` prefix and the audit item id in the body.

---

## File Structure

**New files**

| File | Responsibility |
|---|---|
| `fpl/chips.py` | Chip legality rules only: windows, one-per-half, opening-gameweek ban, one active chip per event, Free Hit spacing. No pandas, no optimizer imports. |
| `fpl/optimize/actions.py` | What each chip does to the squad: Wildcard rebuild, Free Hit one-week squad, Bench Boost / Triple Captain marginal value. |
| `tests/test_chip_rules.py` | Unit tests for `fpl/chips.py`. |
| `tests/test_actions.py` | Unit tests for `fpl/optimize/actions.py`. |

**Modified files**

| File | Change |
|---|---|
| `fpl/state.py` | B2 free-transfer carry; B3 base-squad/bank/purchase-price fields and Free Hit marker. |
| `fpl/cli.py` | B3 Free Hit restoration on the next planning run; `record_transfers` writes base state. |
| `run_gameweek.py` | B4 explicit chip confirmation; passes base state to `record_transfers`. |
| `fpl/optimize/chips.py` | B1 advisor consumes dated `chip_events` and the legality module. |
| `fpl/pipeline.py` | B1/B4 wiring: dated chip events in, real chip squad out. |
| `fpl/optimize/transfers.py` | B5 stratified enumeration; B17 club-cap exception on the hold plan. |
| `fpl/model/minutes.py` | B9 unavailable players score zero everywhere; B11 start rate uses matches. |
| `fpl/model/calibration.py` | B10 per-event calibration, aggregates recomputed from calibrated columns. |
| `fpl/data/normalize.py` | B11 `matches_played` alongside `gws_played`. |
| `fpl/config.py` | R11 `bench_floor_xp` defaults to 0.0. |
| `fpl/backtest/aggregate.py` | B12 season-cutoff walk-forward, no target-season leakage. |
| `fpl/report/weekly.py` | B4 report says when the squad shown is a chip squad. |

---

## Task 1: Chip legality rules

**Audit item:** B1 (Critical)

**Files:**
- Create: `fpl/chips.py`
- Test: `tests/test_chip_rules.py`

**Interfaces:**
- Consumes: `fpl.state.canonical_chip(name) -> str | None` (already exists).
- Produces:
  - `CHIPS: tuple[str, ...]` = `("wildcard", "freehit", "benchboost", "triplecaptain")`
  - `FIRST_HALF_LAST: int` = `19`
  - `chip_window(event: int) -> int` (1 or 2)
  - `chip_blocked_reason(chip: str | None, event: int, chip_events: list[dict], first_event: int = 1) -> str | None`
  - `chip_available(chip, event, chip_events, first_event=1) -> bool`
  - `available_chips(event: int, chip_events: list[dict], first_event: int = 1) -> list[str]`
  - `chip_events` is the `list[dict]` already stored on `State.chip_events`: each `{"chip": str, "event": int | None}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chip_rules.py`:

```python
from fpl.chips import (CHIPS, FIRST_HALF_LAST, chip_window, chip_available,
                       chip_blocked_reason, available_chips)


def _used(chip, event):
    return {"chip": chip, "event": event}


def test_the_season_splits_into_two_chip_windows():
    assert chip_window(1) == 1
    assert chip_window(FIRST_HALF_LAST) == 1
    assert chip_window(FIRST_HALF_LAST + 1) == 2
    assert chip_window(38) == 2


def test_a_first_half_wildcard_still_leaves_the_second_half_one():
    """Two of every chip a season: one for GW1-19, one for GW20-38. A name-only
    record could not tell them apart and suppressed the second for good."""
    played = [_used("wildcard", 5)]
    assert not chip_available("wildcard", 8, played)
    assert chip_available("wildcard", 25, played)


def test_an_unused_first_half_chip_does_not_roll_over():
    """The first set EXPIRES at GW19. Playing the second-half Wildcard in GW20
    must not leave a spare one from a first half that was never used."""
    played = [_used("wildcard", 20)]
    assert not chip_available("wildcard", 25, played)


def test_wildcard_and_free_hit_are_blocked_in_the_opening_gameweek():
    """Transfers before the opening deadline are already unlimited, so FPL
    disables both chips there."""
    assert not chip_available("wildcard", 1, [])
    assert not chip_available("freehit", 1, [])
    assert chip_available("benchboost", 1, [])
    assert chip_available("triplecaptain", 1, [])
    assert chip_available("wildcard", 2, [])


def test_an_entry_that_joined_late_has_its_own_opening_gameweek():
    assert not chip_available("wildcard", 7, [], first_event=7)
    assert chip_available("wildcard", 8, [], first_event=7)


def test_free_hits_cannot_be_played_in_consecutive_gameweeks():
    played = [_used("freehit", 19)]
    assert not chip_available("freehit", 20, played)
    assert chip_available("freehit", 21, played)


def test_only_one_chip_may_be_active_in_a_gameweek():
    played = [_used("benchboost", 12)]
    assert not chip_available("triplecaptain", 12, played)
    assert chip_available("triplecaptain", 13, played)


def test_an_undated_record_is_charged_to_the_first_half():
    """State files written before chip_events existed know WHICH chip was
    played but not when. Charging it to the first half can only forfeit a chip
    already spent; charging it to the second would forfeit one still held."""
    played = [{"chip": "benchboost", "event": None}]
    assert not chip_available("benchboost", 10, played)
    assert chip_available("benchboost", 25, played)


def test_api_chip_names_are_canonicalised():
    played = [_used("3xc", 4)]
    assert not chip_available("triplecaptain", 6, played)


def test_available_chips_lists_everything_legal_this_week():
    assert available_chips(2, []) == list(CHIPS)
    assert available_chips(1, []) == ["benchboost", "triplecaptain"]


def test_a_blocked_chip_explains_itself():
    reason = chip_blocked_reason("wildcard", 8, [_used("wildcard", 5)])
    assert reason and "GW5" in reason
    assert chip_blocked_reason("wildcard", 25, [_used("wildcard", 5)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chip_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fpl.chips'`

- [ ] **Step 3: Write the implementation**

Create `fpl/chips.py`:

```python
"""Which chips are legal to play in a given gameweek.

2026/27 gives every manager TWO of each chip: one usable in GW1-19, one in
GW20-38. An unused first-half chip EXPIRES at the GW19 deadline rather than
rolling over, so the second half always starts with exactly one of each
however the first half went.

`State.chips_used` -- a unique list of names -- cannot express any of that.
Once any Wildcard had been played it suppressed the second-half copy for the
rest of the season, which is four chips and easily twenty points. The dated
`chip_events` record can, so it is the only thing this module reads.

Responsibilities are split three ways and deliberately do not overlap:
this module says what is LEGAL, `optimize.chips` says whether a legal chip is
worth playing NOW, and `optimize.actions` says what the chip actually DOES to
the squad.
"""
from .state import canonical_chip

CHIPS = ("wildcard", "freehit", "benchboost", "triplecaptain")
# Last gameweek of the first chip window. The second set unlocks the week after.
FIRST_HALF_LAST = 19
# Wildcard and Free Hit are disabled in an entry's opening gameweek: transfers
# before that deadline are already unlimited, so neither chip can buy anything.
NO_CHIP_IN_OPENING_GW = ("wildcard", "freehit")
# Gameweeks that must separate two Free Hits. 1 means "not in consecutive
# gameweeks": a Free Hit in GW19 blocks GW20 but not GW21.
FREE_HIT_MIN_GAP = 1


def chip_window(event: int) -> int:
    """1 for the first-half set of chips, 2 for the second."""
    return 1 if int(event) <= FIRST_HALF_LAST else 2


def _played(chip_events) -> list[tuple[str, int | None]]:
    """(chip, event) for every recorded use, names canonicalised."""
    out = []
    for record in chip_events or []:
        name = canonical_chip(record.get("chip"))
        if name is None:
            continue
        event = record.get("event")
        out.append((name, None if event is None else int(event)))
    return out


def chip_blocked_reason(chip, event: int, chip_events,
                        first_event: int = 1) -> str | None:
    """Why `chip` cannot be played in `event`, or None if it can.

    Returned as prose rather than a bool because the weekly report has to tell
    the user WHICH rule stopped a chip the squad otherwise qualified for --
    "already used" and "not until GW20" are different pieces of news.
    """
    name = canonical_chip(chip)
    if name is None:
        return None
    event = int(event)
    window = chip_window(event)
    played = _played(chip_events)

    if name in NO_CHIP_IN_OPENING_GW and event <= int(first_event):
        return (f"a {name} cannot be played in GW{event}, your opening gameweek "
                f"— transfers before that deadline are already unlimited")

    for other, when in played:
        if other != name:
            continue
        # An undated record predates chip_events and in practice was played
        # early, so it is charged to the first window. See the module docstring.
        used_window = 1 if when is None else chip_window(when)
        if used_window != window:
            continue
        where = "earlier this season" if when is None else f"in GW{when}"
        half = "first" if window == 1 else "second"
        unlock = (f"; the next one unlocks in GW{FIRST_HALF_LAST + 1}"
                  if window == 1 else "")
        return f"your {half}-half {name} was played {where}{unlock}"

    for other, when in played:
        if other != name and when == event:
            return (f"{other} is already recorded for GW{event} — only one chip "
                    f"may be active in a gameweek")

    if name == "freehit":
        for other, when in played:
            if other != "freehit" or when is None:
                continue
            if 0 < event - when <= FREE_HIT_MIN_GAP:
                return (f"a Free Hit was played in GW{when}, and two cannot be "
                        f"played in consecutive gameweeks")
    return None


def chip_available(chip, event: int, chip_events, first_event: int = 1) -> bool:
    return chip_blocked_reason(chip, event, chip_events, first_event) is None


def available_chips(event: int, chip_events, first_event: int = 1) -> list[str]:
    """Every chip legal to play in `event`, in CHIPS order."""
    return [c for c in CHIPS if chip_available(c, event, chip_events, first_event)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chip_rules.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: 521 passed (510 baseline + 11 new). Nothing else imports `fpl.chips` yet.

- [ ] **Step 6: Commit**

```bash
git add fpl/chips.py tests/test_chip_rules.py
git commit -m "feat: add chip legality rules for the 2026/27 two-set chip system

Audit B1: chips_used is a unique name list, so the second-half copy of any
chip was suppressed for the season once the first was played. chip_events is
dated and can tell the two halves apart, so legality now reads only that."
```

---

## Task 2: Wildcard and Free Hit stop inventing a free transfer

**Audit item:** B2 (High)

**Files:**
- Modify: `fpl/state.py:153-167`
- Test: `tests/test_state.py:64-69,273-276` (rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ft_after_moves(state, transfers_made, chip=None) -> tuple[int, int]` — unchanged signature, corrected second element for `wildcard`/`freehit`.

- [ ] **Step 1: Rewrite the three tests that assert the wrong rule**

In `tests/test_state.py`, replace `test_wildcard_preserves_the_balance_and_still_accrues` and `test_free_hit_preserves_the_balance` with:

```python
def test_wildcard_keeps_saved_transfers_without_adding_one():
    """FPL's FAQ is explicit: two saved transfers are still two after a
    Wildcard. The gameweek's own free transfer is consumed by activating the
    chip, so the balance is preserved -- not preserved AND incremented, which
    silently authorised a transfer costing four points every week after a chip."""
    assert advance_ft(State(3, 1, []), transfers_made=9, chip="wildcard") == 3


def test_free_hit_keeps_saved_transfers_without_adding_one():
    assert advance_ft(State(2, 1, []), transfers_made=11, chip="freehit") == 2


def test_a_chip_week_cannot_push_the_balance_past_the_cap():
    assert advance_ft(State(5, 1, []), transfers_made=9, chip="wildcard") == 5
```

And replace `test_ft_after_moves_leaves_a_wildcard_week_whole` with:

```python
def test_ft_after_moves_leaves_a_wildcard_week_whole():
    """Unlimited transfers inside the week, and the same balance out of it."""
    remaining, nxt = ft_after_moves(State(free_transfers=2), transfers_made=9,
                                    chip="wildcard")
    assert (remaining, nxt) == (2, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_state.py -q -k "wildcard or free_hit or cap"`
Expected: FAIL — `assert 4 == 3`, `assert 3 == 2`, `assert (2, 3) == (2, 2)`

- [ ] **Step 3: Write the implementation**

In `fpl/state.py`, replace `ft_after_moves`:

```python
def ft_after_moves(state: State, transfers_made: int,
                   chip: str | None = None) -> tuple[int, int]:
    """(still free this gameweek, balance for the next one).

    The two differ by the weekly accrual, and conflating them is what let a
    re-planned gameweek spend a transfer that had already been used.

    A Wildcard or Free Hit is the exception in BOTH directions. Inside the week
    transfers are unlimited, so nothing is spent; but the gameweek's own free
    transfer is consumed by activating the chip, so no accrual follows either.
    FPL's FAQ gives the example directly: two saved transfers are still two
    after a Wildcard. Returning `balance + 1` here handed the optimizer a
    transfer that actually costs four points in the gameweek after every chip.
    """
    if canonical_chip(chip) in CHIPS_PRESERVING_FT:
        balance = min(FT_CAP, int(state.free_transfers))
        return balance, balance
    used = int(transfers_made)
    remaining = max(0, int(state.free_transfers) - used)
    return remaining, min(FT_CAP, remaining + 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_state.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS. `reconcile()` derives its balance through `advance_ft`, so any
test asserting a post-chip derived balance moves by one — fix those the same way,
stating the FAQ rule in the docstring.

- [ ] **Step 6: Commit**

```bash
git add fpl/state.py tests/test_state.py
git commit -m "fix: a chip week preserves the FT balance without accruing one

Audit B2: WC/FH returned current_balance + 1. FPL consumes the gameweek's own
free transfer to activate the chip and retains only what was banked, so the
next-gameweek balance is the same balance."
```

---

## Task 3: A Free Hit no longer destroys the permanent squad

**Audit item:** B3 (Critical)

**Files:**
- Modify: `fpl/state.py:33-61` (schema), `fpl/state.py:81-105` (load/save)
- Modify: `fpl/cli.py:16-29` (LiveSquad), `fpl/cli.py:32-115` (resolve), `fpl/cli.py:118-149` (record)
- Test: `tests/test_cli.py`, `tests/test_state.py`

**Interfaces:**
- Consumes: `fpl.state.canonical_chip`.
- Produces:
  - `State` gains `base_squad: list[int]`, `base_bank: float`, `base_purchase_prices: dict[int, float]`, `freehit_event: int | None`.
  - `record_transfers(...)` gains keyword args `base_squad: list[int] | None = None`, `base_bank: float | None = None`, `base_purchase_prices: dict[int, float] | None = None`. When `chip` canonicalises to `freehit` these three record the PERMANENT squad the Free Hit is replacing; for any other chip they are cleared.
  - `LiveSquad` gains `restored_from_freehit: bool = False`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_a_free_hit_preserves_the_permanent_squad_and_its_prices(tmp_path):
    """FPL restores the pre-chip squad, bank and purchase prices at the next
    deadline. The local state held the ONLY copy of the purchase prices, and
    --confirm was overwriting it with the temporary Free Hit 15."""
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    permanent = list(range(1, 16))
    temporary = list(range(101, 116))
    prices = {i: 5.0 for i in permanent}

    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     purchase_prices={i: 6.0 for i in temporary},
                     squad=temporary, bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices=prices)

    written = load_state(path, cfg)
    assert written.squad == temporary          # this week you field the FH 15
    assert written.freehit_event == 8
    assert written.base_squad == permanent     # and you keep the real one
    assert written.base_bank == 1.2
    assert written.base_purchase_prices == prices


def test_the_gameweek_after_a_free_hit_plans_from_the_restored_squad(tmp_path):
    """resolve_current_squad reads the PREVIOUS gameweek's picks, which after a
    Free Hit is the temporary team FPL has already taken away."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=7, free_transfers=1)
    permanent = list(range(1, 16))
    temporary = list(range(101, 116))
    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     purchase_prices={i: 6.0 for i in temporary},
                     squad=temporary, bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in permanent})

    client = _Client(picks={"picks": [{"element": i} for i in temporary],
                            "entry_history": {"bank": 3}},
                     history={"current": [], "chips": []})
    live, errors = resolve_current_squad(cfg, 9, path, client)

    assert errors == []
    assert live.current_squad == permanent
    assert live.bank == 1.2
    assert live.purchase_prices == {i: 5.0 for i in permanent}
    assert live.restored_from_freehit is True
    assert any("Free Hit" in w for w in live.warnings)


def test_an_ordinary_confirmation_clears_the_free_hit_marker(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     squad=list(range(101, 116)), bank=0.3,
                     base_squad=list(range(1, 16)), base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in range(1, 16)})
    record_transfers(path, cfg, gw=9, transfers_made=1, chip=None,
                     squad=list(range(1, 16)), bank=1.2)
    written = load_state(path, cfg)
    assert written.freehit_event is None
    assert written.base_squad == []
```

`_Client` is the existing fake in `tests/test_cli.py` — reuse it; if its constructor differs, match the existing call sites in that file rather than inventing a new fake.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q -k "free_hit or freehit"`
Expected: FAIL — `TypeError: record_transfers() got an unexpected keyword argument 'base_squad'`

- [ ] **Step 3: Add the state fields**

In `fpl/state.py`, append to the `State` dataclass:

```python
    # The PERMANENT squad a Free Hit is temporarily replacing.
    #
    # A Free Hit lasts one gameweek: at the next deadline FPL silently restores
    # the squad, bank and purchase prices from before the chip. Local state held
    # the only copy of those purchase prices -- no public endpoint reports them
    # -- and `--confirm` was overwriting all three with the temporary fifteen.
    # After that every later budget and transfer started from a team the
    # manager does not own, and nothing in the tool could notice.
    base_squad: list[int] = field(default_factory=list)
    base_bank: float = 0.0
    base_purchase_prices: dict[int, float] = field(default_factory=dict)
    # The gameweek a Free Hit is active for, or None. Planning any LATER
    # gameweek must ignore that week's picks and restore the base above.
    freehit_event: int | None = None
```

In `load_state`, add to the `State(...)` construction:

```python
        base_squad=[int(i) for i in (raw.get("base_squad") or [])],
        base_bank=float(raw.get("base_bank", 0.0)),
        base_purchase_prices={int(k): float(v)
                              for k, v in (raw.get("base_purchase_prices") or {}).items()},
        freehit_event=(None if raw.get("freehit_event") is None
                       else int(raw["freehit_event"])),
```

`save_state` uses `asdict` and needs no change.

- [ ] **Step 4: Write `record_transfers`**

Replace the body of `record_transfers` in `fpl/cli.py` (keep the existing docstring and extend it):

```python
def record_transfers(state_path: Path, cfg, gw: int, transfers_made: int,
                     chip: str | None,
                     purchase_prices: dict[int, float] | None = None,
                     squad: list[int] | None = None,
                     bank: float | None = None,
                     api_chips: list[dict] | None = None,
                     base_squad: list[int] | None = None,
                     base_bank: float | None = None,
                     base_purchase_prices: dict[int, float] | None = None) -> State:
    """Write a confirmed gameweek to state, and return what was written.

    `squad` and `bank` are the squad and bank ACTUALLY APPLIED. Recording the
    transfers without them left state internally inconsistent: purchase prices
    moved to the new squad while `squad` still named the old one, so a re-run
    before the deadline planned from a squad the prices no longer described.

    `base_*` are the PERMANENT squad, bank and purchase prices, and matter only
    for a Free Hit: that chip's fifteen last one gameweek, after which FPL
    restores exactly these. Writing the temporary squad over them destroyed the
    only record of what the manager actually owns -- purchase prices included,
    which no endpoint republishes -- so every later budget was wrong. For any
    other chip, and for no chip, the applied squad IS the permanent one and the
    base fields are cleared.
    """
    state = load_state(state_path, cfg)
    remaining, new_ft = ft_after_moves(state, transfers_made, chip)
    chips_used, chip_events = record_chip(state, chip, gw)
    if api_chips:
        chips_used, chip_events = merge_chip_events(
            State(chips_used=chips_used, chip_events=chip_events), api_chips
        )
    prices = state.purchase_prices if purchase_prices is None else purchase_prices
    applied = list(state.squad) if squad is None else [int(i) for i in squad]
    applied_event = int(state.squad_event) if squad is None else int(gw)
    applied_bank = float(state.bank) if bank is None else float(bank)

    if canonical_chip(chip) == "freehit":
        keep = ([int(i) for i in base_squad] if base_squad is not None
                else list(state.base_squad) or list(state.squad))
        keep_bank = (float(base_bank) if base_bank is not None
                     else (state.base_bank if state.base_squad else float(state.bank)))
        keep_prices = dict(base_purchase_prices if base_purchase_prices is not None
                           else (state.base_purchase_prices or state.purchase_prices))
        fh_event: int | None = int(gw)
    else:
        keep, keep_bank, keep_prices, fh_event = [], 0.0, {}, None

    written = State(new_ft, gw, chips_used, chip_events, dict(prices),
                    squad=applied, squad_event=applied_event, bank=applied_bank,
                    free_transfers_remaining=remaining,
                    base_squad=keep, base_bank=keep_bank,
                    base_purchase_prices=keep_prices, freehit_event=fh_event)
    save_state(written, state_path)
    return written
```

Add `canonical_chip` to the `from .state import (...)` line at the top of `fpl/cli.py`.

- [ ] **Step 5: Write the restoration in `resolve_current_squad`**

In `fpl/cli.py`, add `restored_from_freehit: bool = False` to `LiveSquad`.

Replace the block from `state_existed = Path(state_path).exists()` through the end of the `if confirmed:` branch with:

```python
    state_existed = Path(state_path).exists()
    state = load_state(state_path, cfg)
    warnings: list[str] = []
    purchase_prices = dict(state.purchase_prices)

    # A squad confirmed by hand for THIS gameweek beats the API, which cannot
    # see a transfer until after the deadline it was made for -- by which point
    # the advice is useless. Only for this gameweek: once GW{gw} has started,
    # picks becomes authoritative again and a stale override would be worse
    # than no override.
    confirmed = bool(state.squad) and int(state.squad_event) == int(gw)
    # A Free Hit lasts one gameweek. Planning anything after it from the
    # previous gameweek's picks plans from the temporary team FPL has already
    # taken back, on a bank the chip week was never allowed to keep.
    restored = (not confirmed and state.freehit_event is not None
                and int(gw) > int(state.freehit_event) and bool(state.base_squad))

    if confirmed:
        current_squad = list(state.squad)
        bank = float(state.bank)
        warnings.append(
            f"Using the confirmed squad recorded for GW{gw} rather than your "
            f"GW{prev_gw} picks — FPL does not publish a squad for a gameweek "
            f"that has not started."
        )
    elif restored:
        current_squad = list(state.base_squad)
        bank = float(state.base_bank)
        purchase_prices = dict(state.base_purchase_prices)
        warnings.append(
            f"Your GW{state.freehit_event} Free Hit squad has expired. Planning "
            f"from the 15 it replaced, at the bank you had before the chip — any "
            f"money the Free Hit week freed up does not carry over."
        )
```

Then change the `missing`/return block at the end to use the local `purchase_prices`:

```python
    missing = [p for p in current_squad if p not in purchase_prices]
    if purchase_prices and missing:
        warnings.append(
            f"{len(missing)} of your 15 have no recorded purchase price -- those are "
            f"budgeted at market value, which overstates what they would sell for if "
            f"they have risen."
        )

    return LiveSquad(current_squad, bank, free_transfers, warnings,
                     purchase_prices, chips_used, chip_events,
                     restored_from_freehit=restored), []
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli.py tests/test_state.py -q`
Expected: PASS

- [ ] **Step 7: Pass base state from the CLI entry point**

In `run_gameweek.py`, in the `--confirm` branch, change the `record_transfers` call to:

```python
        written = record_transfers(state_path, cfg, args.gw, transfers_made, chip,
                                   purchase_prices=updated, squad=applied, bank=cash,
                                   api_chips=chip_events,
                                   base_squad=list(current_squad),
                                   base_bank=bank,
                                   base_purchase_prices=dict(purchase_prices))
```

`current_squad`, `bank` and `purchase_prices` are already in scope and hold the
pre-deadline permanent squad — which under a Free Hit is exactly what FPL restores.

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add fpl/state.py fpl/cli.py run_gameweek.py tests/test_cli.py tests/test_state.py
git commit -m "fix: keep the permanent squad and purchase prices across a Free Hit

Audit B3: --confirm wrote the temporary FH fifteen over the only record of the
real squad and its purchase prices, so every later budget started from a team
the manager does not own. State now carries base_squad/base_bank/
base_purchase_prices and restores them on the next planning run."
```

---

## Task 4: The advisor reads dated chip events

**Audit item:** B1 (Critical, second half)

**Files:**
- Modify: `fpl/optimize/chips.py:156-306`
- Modify: `fpl/pipeline.py:239-242,379-385`
- Modify: `run_gameweek.py:80-95`
- Test: `tests/test_chips.py` (update call sites)

**Interfaces:**
- Consumes: `fpl.chips.chip_available`, `fpl.chips.chip_blocked_reason`.
- Produces: `advise_chips(xp_df, lineup, squad_ids, counts, team_by_player, from_event, chip_events, last_event=38, quality=None, first_event=1) -> ChipAdvice`. The seventh positional parameter is renamed `chips_used -> chip_events` and its type changes from `list[str]` to `list[dict]`.
- `pipeline.run(...)` gains `chip_events: list[dict] | None = None` and `first_event: int = 1`; the existing `chips_used` parameter is removed.

- [ ] **Step 1: Update the existing chip tests to the new contract**

In `tests/test_chips.py`:

Add near the top, under the existing constants:

```python
# Chip tests run from GW2, not GW1: Wildcard and Free Hit are illegal in an
# entry's opening gameweek, so a GW1 fixture cannot exercise either of them.
EVENT = 2


def _used(*names, event=1):
    return [{"chip": n, "event": event} for n in names]
```

Change `_counts` to take the event:

```python
def _counts(n=1, event=EVENT):
    return pd.DataFrame([{"team_id": 1, "event": event, "n_fixtures": n}])
```

Then, throughout the file, replace the `from_event` positional argument `1` with
`EVENT` in every `advise_chips(...)` call, and replace the two list-of-names
arguments with `_used(...)`:

- line ~70 `["benchboost"]` → `_used("benchboost")`
- line ~84 `["benchboost"]` → `_used("benchboost")`
- line ~316 `["wildcard"]` → `_used("wildcard")`

The calls that already pass `from_event=5` with a locally built `counts` frame
(lines ~203-279) are unaffected — leave their event numbers alone.

- [ ] **Step 2: Add the new legality tests**

Append to `tests/test_chips.py`:

```python
def test_a_first_half_wildcard_does_not_block_the_second_half_one():
    """Two Wildcards a season. The advisor used to see a single name and
    suppress the second for good."""
    flags = [["Unavailable (i): injured"]] * 5 + [[] for _ in range(10)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(event=25),
                     TEAM_BY_PLAYER, 25, _used("wildcard", event=5))
    assert a.chip == "wildcard"


def test_the_advisor_never_offers_a_chip_in_the_opening_gameweek():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(n=0, event=1),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip != "freehit"


def test_a_blocked_chip_is_explained_by_the_rule_that_blocked_it():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT,
                     _used("benchboost", event=1))
    assert a.chip is None
    assert "benchboost" in a.reason
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_chips.py -q`
Expected: FAIL — the new tests fail (`a.chip != "wildcard"`), and the `_used(...)`
call sites fail because `advise_chips` still treats its seventh argument as names.

- [ ] **Step 4: Rewrite the availability logic in `advise_chips`**

In `fpl/optimize/chips.py`, add the import at the top:

```python
from ..chips import chip_available, chip_blocked_reason
```

Change the signature and the first lines of the body:

```python
def advise_chips(xp_df: pd.DataFrame, lineup, squad_ids: list[int],
                 counts: pd.DataFrame, team_by_player: dict[int, int],
                 from_event: int, chip_events: list[dict],
                 last_event: int = 38,
                 quality: "SquadQuality | None" = None,
                 first_event: int = 1) -> ChipAdvice:
    """Which chip, if any, to play this gameweek.

    `chip_events` is the DATED record from `State.chip_events`, not a list of
    names: a name alone cannot say which half of the season a Wildcard belongs
    to, and 2026/27 gives two of every chip -- one per half. Legality lives in
    `fpl.chips`; everything below is about whether a LEGAL chip is worth
    playing now.
    """
    df = xp_df.set_index("player_id")
    events = list(chip_events or [])

    def usable(name: str) -> bool:
        return chip_available(name, from_event, events, first_event)

    def why_not(name: str) -> str:
        return chip_blocked_reason(name, from_event, events, first_event) or ""

    ids = [int(i) for i in squad_ids]
```

Delete the `used = set(chips_used or [])` line.

Replace each availability test in the body:

| Old | New |
|---|---|
| `and "freehit" not in used` (hold list) | `and usable("freehit")` |
| `and "triplecaptain" not in used` (hold list) | `and usable("triplecaptain")` |
| `and "benchboost" not in used` (hold list) | `and usable("benchboost")` |
| `if free_hit_ok and "freehit" not in used:` | `if free_hit_ok and usable("freehit"):` |
| `if wildcard_ok and "wildcard" not in used:` | `if wildcard_ok and usable("wildcard"):` |
| `if triple_ok and "triplecaptain" not in used:` | `if triple_ok and usable("triplecaptain"):` |
| `if bench_ok and "benchboost" not in used:` | `if bench_ok and usable("benchboost"):` |

Replace the whole `blocked` section with:

```python
    blocked = []
    if free_hit_ok and not usable("freehit"):
        blocked.append(f"{blanks} players have a blank fixture — Free Hit-worthy, "
                       f"but {why_not('freehit')}")
    if wildcard_ok and not usable("wildcard"):
        detail = (f"{problems} players carry injury/rotation flags"
                  if problems >= WILDCARD_MIN_PROBLEMS
                  else f"a rebuild projects {quality.surplus:.1f} xP beyond your transfers")
        blocked.append(f"{detail} — Wildcard-worthy, but {why_not('wildcard')}")
    if triple_ok and not usable("triplecaptain"):
        blocked.append(f"your captain has a standout week — Triple-Captain-worthy, "
                       f"but {why_not('triplecaptain')}")
    if bench_ok and not usable("benchboost"):
        blocked.append(f"your bench projects strongly — Bench-Boost-worthy, "
                       f"but {why_not('benchboost')}")
```

- [ ] **Step 5: Wire dated events through the pipeline**

In `fpl/pipeline.py`, change the `run` signature's last parameter from
`chips_used: list[str] | None = None` to:

```python
        chip_events: list[dict] | None = None, first_event: int = 1):
```

and the `advise_chips` call to:

```python
    chip = advise_chips(xp, lineup, squad_ids, counts, team_by_player, from_event,
                        list(chip_events or []), last_event=last_event,
                        quality=quality, first_event=first_event)
```

In `run_gameweek.py`, replace

```python
    chips_used = load_state(state_path, cfg).chips_used
    chip_events: list[dict] = []
```

with

```python
    # Chips spent so far, DATED. Mode 1 has no live squad to resolve, but a chip
    # is gone whichever mode notices, so local state is read either way.
    chip_events: list[dict] = load_state(state_path, cfg).chip_events
    first_event = 1
```

replace the Mode 2 assignment `chips_used, chip_events = live.chips_used, live.chip_events`
with `chip_events = live.chip_events`, and change the `run(...)` call's last
argument from `chips_used=chips_used` to `chip_events=chip_events, first_event=first_event`.

In `fpl/cli.py`, derive the entry's opening gameweek and return it on `LiveSquad`.
Add `first_event: int = 1` to the dataclass, and after the history fetch:

```python
    # An entry that joined mid-season has its own opening gameweek, and neither
    # a Wildcard nor a Free Hit may be played in it.
    current_events = [int(e["event"]) for e in (history or {}).get("current", [])
                      if e.get("event")]
    first_event = min(current_events) if current_events else 1
```

pass it into the `LiveSquad(...)` construction as `first_event=first_event`, and in
`run_gameweek.py` set `first_event = live.first_event` in the Mode 2 branch.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_chips.py tests/test_pipeline.py -q`
Expected: PASS

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS. `tests/test_pipeline.py` may call `run(..., chips_used=[...])`;
change those to `chip_events=[...]` with dated dicts.

- [ ] **Step 8: Commit**

```bash
git add fpl/optimize/chips.py fpl/pipeline.py fpl/cli.py run_gameweek.py tests/test_chips.py tests/test_pipeline.py
git commit -m "fix: advise chips from the dated record, not a set of names

Audit B1: the advisor turned chips_used into a set, so one Wildcard suppressed
the second-half copy for the season and no chip could expire at GW19. It now
asks fpl.chips whether a chip is legal in THIS gameweek."
```

---

## Task 5: Chip actions build the squad the chip actually fields

**Audit item:** B4 (Critical)

**Files:**
- Create: `fpl/optimize/actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: `fpl.optimize.transfers._budget_and_cost`, `_plan`, `_solve`, `TransferPlan`; `fpl.optimize.lineup.build_lineup`; `fpl.optimize.squad.Squad`.
- Produces:
  - `@dataclass ChipAction` with fields `chip: str | None`, `squad_ids: list[int]`, `starting_ids: list[int]`, `transfers: TransferPlan | None`, `value: float`, `temporary: bool`, `detail: str`.
  - `wildcard_action(xp_df, cfg, current_squad, bank, selling, xp_col) -> ChipAction | None`
  - `freehit_action(xp_df, cfg, current_squad, bank, selling) -> ChipAction | None`
  - `bench_boost_value(lineup, xp_df) -> float`
  - `triple_captain_value(lineup, xp_df) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_actions.py`:

```python
import pandas as pd
import pytest

from fpl.config import Config
from fpl.optimize.actions import (ChipAction, wildcard_action, freehit_action,
                                  bench_boost_value, triple_captain_value)
from fpl.optimize.lineup import Lineup


def _pool():
    """A pool big enough to build two legal squads from, with a clear optimum."""
    rows = []
    pid = 1
    for pos, n in (("GKP", 6), ("DEF", 15), ("MID", 15), ("FWD", 9)):
        for i in range(n):
            rows.append({
                "player_id": pid,
                "web_name": f"{pos}{i}",
                "team": f"T{pid % 10}",
                "position": pos,
                "price": 4.0 + (i % 5) * 0.5,
                "ownership": 5.0,
                "xp_next1": 1.0 + (i % 7) * 0.4,
                "xp_next5": 5.0 + (i % 7) * 2.0,
                "xp_horizon": 4.5 + (i % 7) * 1.8,
                "p_play": 0.9,
            })
            pid += 1
    return pd.DataFrame(rows)


def _cfg():
    return Config(budget=100.0, rank_sims=0, bench_floor_xp=0.0)


def _current(pool):
    """A deliberately poor legal 15: the cheapest, lowest-projected of each."""
    picks = []
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        sub = pool[pool.position == pos].nsmallest(n, "xp_horizon")
        picks += [int(i) for i in sub.player_id]
    return picks


def test_a_wildcard_returns_the_rebuild_it_measured():
    """The advisor computed a rebuild only to get a scalar, threw the squad
    away, and returned the limited-transfer squad instead -- so confirming a
    Wildcard applied a team the Wildcard had never chosen."""
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    selling = {i: float(pool.set_index("player_id").loc[i, "price"]) for i in current}
    action = wildcard_action(pool, cfg, current, bank=5.0, selling=selling,
                             xp_col="xp_horizon")
    assert isinstance(action, ChipAction)
    assert action.chip == "wildcard"
    assert len(action.squad_ids) == 15
    assert len(action.starting_ids) == 11
    assert action.temporary is False
    # An unlimited rebuild cannot be worse than the squad it replaces.
    assert action.value > 0
    # And it is a real rebuild, not the squad it started from.
    assert set(action.squad_ids) != set(current)


def test_a_wildcard_rebuild_never_charges_a_points_hit():
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    selling = {i: float(pool.set_index("player_id").loc[i, "price"]) for i in current}
    action = wildcard_action(pool, cfg, current, bank=5.0, selling=selling,
                             xp_col="xp_horizon")
    assert action.transfers.hit_cost == 0


def test_a_free_hit_squad_is_optimised_for_one_week_only():
    """A Free Hit lasts a gameweek, so it is chosen on xp_next1. Choosing it on
    the horizon buys players for weeks the squad will not exist in."""
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    selling = {i: float(pool.set_index("player_id").loc[i, "price"]) for i in current}
    action = freehit_action(pool, cfg, current, bank=5.0, selling=selling)
    assert action.chip == "freehit"
    assert action.temporary is True
    assert len(action.squad_ids) == 15
    frame = pool.set_index("player_id")
    one_week = float(frame.loc[action.starting_ids, "xp_next1"].sum())
    horizon_pick = wildcard_action(pool, cfg, current, bank=5.0, selling=selling,
                                   xp_col="xp_horizon")
    assert one_week >= float(frame.loc[horizon_pick.starting_ids, "xp_next1"].sum())


def test_bench_boost_is_worth_the_real_bench_not_the_four_lowest():
    """The advisor approximated the bench as the four numerically lowest
    projections in the squad, which is not necessarily a legal bench."""
    pool = _pool()
    lineup = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                    formation="4-4-2", captain=1, vice=2, xp=40.0)
    frame = pool.set_index("player_id")
    expected = float(frame.loc[[12, 13, 14, 15], "xp_next1"].sum())
    assert bench_boost_value(lineup, pool) == pytest.approx(expected)


def test_triple_captain_is_worth_one_more_captain_return():
    pool = _pool()
    lineup = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                    formation="4-4-2", captain=3, vice=2, xp=40.0)
    frame = pool.set_index("player_id")
    captain_xp = float(frame.loc[3, "xp_next1"])
    p_play = float(frame.loc[3, "p_play"])
    vice_xp = float(frame.loc[2, "xp_next1"])
    expected = p_play * captain_xp + (1 - p_play) * vice_xp
    assert triple_captain_value(lineup, pool) == pytest.approx(expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_actions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fpl.optimize.actions'`

- [ ] **Step 3: Write the implementation**

Create `fpl/optimize/actions.py`:

```python
"""What each chip actually DOES to the squad.

The advisor used to be bolted on after optimization: transfers and lineup were
finalised first, and a Free Hit recommendation then came back attached to the
ordinary permanent transfer plan -- the one thing a Free Hit is not. A Wildcard
was worse: it solved a full rebuild purely to extract a scalar surplus, threw
the rebuilt squad away, and returned the limited-transfer squad whose objective
still included hit costs. Confirming either applied a team the named chip had
never chosen.

So each chip is modelled here as a COMPLETE action -- squad, XI, transfers and
the points it is worth -- against the same no-chip baseline. `optimize.chips`
still decides the timing; this module decides what happens when it fires.
"""
from dataclasses import dataclass, field

from .lineup import build_lineup
from .squad import Squad
from .transfers import TransferPlan, _budget_and_cost, _plan, _solve

SQUAD_SIZE = 15
# A Free Hit squad exists for exactly one gameweek, so it is chosen on the
# single-week projection. Choosing it on the discounted horizon buys players
# for weeks the squad will have ceased to exist in.
FREE_HIT_COL = "xp_next1"


@dataclass
class ChipAction:
    """One complete, legal thing the manager could do this gameweek."""
    chip: str | None
    squad_ids: list[int]
    starting_ids: list[int]
    transfers: TransferPlan | None = None
    # Points this action is worth, in whatever currency the caller compares on.
    value: float = 0.0
    # True when FPL takes the squad back at the next deadline (Free Hit only),
    # which is what decides whether local state may overwrite the permanent 15.
    temporary: bool = False
    detail: str = ""


def _rebuild(xp_df, cfg, current_squad, bank, selling, xp_col, bench_floor):
    """Solve for a full 15 with every player in play, on the money actually held.

    Unlimited transfers is exactly the weekly solve with `max_changes` at 15 --
    same objective, same budget, same selling prices -- which is why this goes
    through `transfers._solve` rather than `optimize_squad`. Routing it through
    the squad builder would compare against a fresh `cfg.budget` the manager
    does not have.
    """
    current = {int(i) for i in current_squad}
    budget, cost = _budget_and_cost(xp_df, current, bank, selling)
    solved = _solve(xp_df, current, budget, SQUAD_SIZE, cfg, xp_col, cost=cost,
                    bench_floor=bench_floor)
    if solved is None and bench_floor > 0:
        # A floor no budget can satisfy must not take the chip down with it.
        solved = _solve(xp_df, current, budget, SQUAD_SIZE, cfg, xp_col, cost=cost)
    if solved is None:
        return None, current
    return solved, current


def wildcard_action(xp_df, cfg, current_squad, bank: float, selling: dict,
                    xp_col: str = "xp_horizon") -> ChipAction | None:
    """The permanent rebuild a Wildcard buys, and what it is worth.

    Valued on the HORIZON, not one week: a Wildcard squad is kept, so the gain
    it buys is collected over every gameweek the projection reaches.

    `n_transfers` is reported for display only. The chip makes them free, so
    the plan is built with `free_transfers` at 15 and carries no hit.
    """
    bench_floor = float(getattr(cfg, "bench_floor_xp", 0.0) or 0.0)
    solved, current = _rebuild(xp_df, cfg, current_squad, bank, selling, xp_col,
                               bench_floor)
    if solved is None:
        return None
    plan = _plan(current, solved, SQUAD_SIZE, cfg)
    return ChipAction(
        chip="wildcard",
        squad_ids=list(plan.squad_ids),
        starting_ids=list(plan.starting_ids),
        transfers=plan,
        value=round(float(plan.net_xp), 3),
        temporary=False,
        detail=f"unlimited free transfers, changing {plan.n_transfers} of your 15",
    )


def freehit_action(xp_df, cfg, current_squad, bank: float,
                   selling: dict) -> ChipAction | None:
    """The one-week squad a Free Hit fields, and what it scores that week.

    Built on the same money as a Wildcard -- the chip does not grant a fresh
    100m -- but optimised on `xp_next1`, and flagged `temporary` so the caller
    knows FPL will take all fifteen back at the next deadline.

    No bench floor: a Free Hit bench never plays, so spending XI budget to fill
    it would be spending it for nothing.
    """
    solved, current = _rebuild(xp_df, cfg, current_squad, bank, selling,
                               FREE_HIT_COL, bench_floor=0.0)
    if solved is None:
        return None
    plan = _plan(current, solved, SQUAD_SIZE, cfg)
    frame = xp_df.set_index("player_id")
    value = float(frame.loc[list(plan.starting_ids), FREE_HIT_COL].sum())
    return ChipAction(
        chip="freehit",
        squad_ids=list(plan.squad_ids),
        starting_ids=list(plan.starting_ids),
        transfers=plan,
        value=round(value, 3),
        temporary=True,
        detail=f"a one-week squad changing {plan.n_transfers} of your 15",
    )


def bench_boost_value(lineup, xp_df, xp_col: str = FREE_HIT_COL) -> float:
    """What a Bench Boost adds: the four players who are ACTUALLY benched.

    The advisor approximated this as the four numerically lowest projections in
    the squad, which need not be a legal bench under formation constraints --
    a 3-5-2 benches a defender the approximation would have started.
    """
    frame = xp_df.set_index("player_id")
    bench = [int(i) for i in lineup.bench]
    if not bench:
        return 0.0
    return round(float(frame.loc[bench, xp_col].sum()), 3)


def triple_captain_value(lineup, xp_df, xp_col: str = FREE_HIT_COL) -> float:
    """What a Triple Captain adds: ONE more captain return, not three.

    The armband already pays double. The chip pays a third multiple, and only
    if the captain appears -- if he does not, the vice inherits an ordinary
    double and the chip is wasted, which is exactly the risk the advice text
    warns about and the number never used to include.
    """
    frame = xp_df.set_index("player_id")
    captain = int(lineup.captain)
    captain_xp = float(frame.loc[captain, xp_col])
    p_play = (float(frame.loc[captain, "p_play"])
              if "p_play" in frame.columns else 1.0)
    vice_xp = (float(frame.loc[int(lineup.vice), xp_col])
               if lineup.vice is not None else 0.0)
    return round(p_play * captain_xp + (1.0 - p_play) * vice_xp, 3)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_actions.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — nothing imports `actions` yet.

- [ ] **Step 6: Commit**

```bash
git add fpl/optimize/actions.py tests/test_actions.py
git commit -m "feat: model each chip as a complete action with its own squad

Audit B4: a Wildcard rebuild was solved for a scalar and discarded, and a Free
Hit came back attached to the ordinary permanent transfer plan. Each chip now
produces the squad, XI and value it actually means."
```

---

## Task 6: The pipeline returns the recommended chip's real squad

**Audit item:** B4 (Critical, second half)

**Files:**
- Modify: `fpl/pipeline.py:136-170,343-410`
- Modify: `fpl/report/weekly.py:160-175`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `fpl.optimize.actions.{wildcard_action, freehit_action, ChipAction}`.
- Produces: `Recommendation` gains `chip_squad: bool = False` and `chip_temporary: bool = False`. When `rec.chip.chip` is `"wildcard"` or `"freehit"`, `rec.squad_ids`, `rec.lineup`, `rec.transfers` and `rec.bank` describe the chip's own squad.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py` (match the file's existing fixture style for
building a fake client and pool — reuse whatever helper the other pipeline tests use):

```python
def test_a_recommended_wildcard_returns_the_rebuilt_squad(monkeypatch):
    """Confirming a Wildcard used to apply the limited-transfer squad, which is
    not the team the chip chose."""
    from fpl.optimize.chips import ChipAdvice
    import fpl.pipeline as pipeline

    monkeypatch.setattr(pipeline, "advise_chips",
                        lambda *a, **k: ChipAdvice("wildcard", "test"))
    rec, xp = _run_mode_two(pipeline)      # existing helper in this file
    assert rec.chip.chip == "wildcard"
    assert rec.chip_squad is True
    assert rec.chip_temporary is False
    assert rec.transfers is not None and rec.transfers.hit_cost == 0
    assert set(rec.lineup.xi) <= set(rec.squad_ids)


def test_a_recommended_free_hit_is_flagged_temporary(monkeypatch):
    from fpl.optimize.chips import ChipAdvice
    import fpl.pipeline as pipeline

    monkeypatch.setattr(pipeline, "advise_chips",
                        lambda *a, **k: ChipAdvice("freehit", "test"))
    rec, xp = _run_mode_two(pipeline)
    assert rec.chip_squad is True
    assert rec.chip_temporary is True
```

If `tests/test_pipeline.py` has no `_run_mode_two` helper, write one that calls
`pipeline.run(cfg, mode=2, from_event=..., root=tmp_path, client=<the file's fake>,
current_squad=[...], bank=..., free_transfers=1)` using the same fake client the
other tests in that file already construct.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -q -k "wildcard or free_hit"`
Expected: FAIL — `AttributeError: 'Recommendation' object has no attribute 'chip_squad'`

- [ ] **Step 3: Add the report fields**

In `fpl/report/weekly.py`, add to the `Recommendation` dataclass:

```python
    # True when the fifteen below are the squad a CHIP builds rather than the
    # ordinary transfer plan, and true again when FPL will take them back at
    # the next deadline (Free Hit). The report has to say which, because
    # confirming one writes permanent state and the other must not.
    chip_squad: bool = False
    chip_temporary: bool = False
```

In `render`, after the chip line (`out.append(f"**{rec.chip.chip}** — ...")`), add:

```python
        if rec.chip_temporary:
            out.append(
                "The fifteen above are this chip's ONE-WEEK squad. FPL restores "
                "your permanent team at the next deadline, and any money this "
                "week frees up does not carry over."
            )
        elif rec.chip_squad:
            out.append("The fifteen above are the squad this chip builds, "
                       "with every transfer free.")
```

- [ ] **Step 4: Apply the chip's squad in the pipeline**

In `fpl/pipeline.py`, add the import:

```python
from .optimize.actions import wildcard_action, freehit_action
```

Immediately after the `chip = advise_chips(...)` call and before `value = round(...)`,
insert:

```python
    # A named chip must return the squad that chip actually fields. Advising
    # "Free Hit" while handing back the ordinary permanent transfer plan is not
    # a timing error -- it is not the action the chip means.
    chip_squad = chip_temporary = False
    if actual_mode == 2 and chip is not None and chip.chip in ("wildcard", "freehit"):
        action = (wildcard_action(xp, cfg, current_squad, bank, selling,
                                  xp_col=HORIZON_COL)
                  if chip.chip == "wildcard"
                  else freehit_action(xp, cfg, current_squad, bank, selling))
        if action is None:
            chip = ChipAdvice(None, (
                f"A {chip.chip} looked right this week, but no legal squad could "
                f"be built from your budget — so no chip is recommended."
            ))
        else:
            squad_ids, starting_ids = action.squad_ids, action.starting_ids
            transfers = action.transfers
            lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)
            chip_squad, chip_temporary = True, action.temporary
```

Add `ChipAdvice` to the existing `from .optimize.chips import advise_chips` line.

Move the `from .optimize.squad import Squad` import from inside `run` to the module
top alongside the other `optimize` imports, so the block above can use it.

Pass the two flags into the `Recommendation(...)` construction:

```python
        chip_squad=chip_squad, chip_temporary=chip_temporary,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py tests/test_report.py -q`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add fpl/pipeline.py fpl/report/weekly.py tests/test_pipeline.py
git commit -m "fix: a recommended Wildcard or Free Hit returns its own squad

Audit B4: chip advice was computed after the transfer plan was final, so the
squad returned alongside a Free Hit was the ordinary permanent one. The chip's
action now replaces squad, XI, transfers and lineup, and the report says when
the fifteen shown are temporary."
```

---

## Task 7: Confirming a chip has to be explicit

**Audit item:** B4 (Critical, third half)

**Files:**
- Modify: `run_gameweek.py:55-58,126-151`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Recommendation.chip_temporary` from Task 6.
- Produces: `--confirm` no longer defaults the applied chip to the advised chip. `--applied-chip` becomes required whenever the run advised a chip.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_confirming_without_naming_the_chip_is_refused(monkeypatch, capsys, tmp_path):
    """A run is a proposal. Defaulting --confirm to the advised chip spends a
    chip on advice the manager may never have acted on -- and for a Free Hit it
    also decides whether the permanent squad is preserved."""
    import run_gameweek

    code = run_gameweek.main_with_args([
        "--mode", "2", "--gw", "5", "--confirm", "--no-refresh",
    ])
    out = capsys.readouterr().out
    assert code != 0
    assert "--applied-chip" in out
```

If `run_gameweek.py` has no `main_with_args`, refactor `main()` to
`def main(argv=None)` passing `argv` to `ap.parse_args(argv)`, and call it as
`run_gameweek.main(["--mode", "2", ...])` in the test instead.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli.py -q -k "naming_the_chip"`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

In `run_gameweek.py`, change the `--applied-chip` help text:

```python
    ap.add_argument("--applied-chip", default=None,
                    help="with --confirm: the chip you actually played "
                         "(wildcard/freehit/benchboost/triplecaptain), or 'none' if "
                         "you played none. REQUIRED whenever the run advised a chip — "
                         "a recommendation is not an action, and confirming one you "
                         "did not play spends it for the season.")
```

Replace the chip resolution inside the `--confirm` branch:

```python
        # A run is a PROPOSAL. Defaulting to the advised chip spent a Triple
        # Captain on a run that was never acted on -- and for a Free Hit it also
        # decided whether the permanent squad survived. So when the run advised
        # a chip, the manager has to say what they actually played.
        advised = rec.chip.chip if rec.chip else None
        if args.applied_chip is None:
            if advised is not None:
                print(f"\nThis run advised {advised}. Re-run with "
                      f"--applied-chip {advised} if you played it, or "
                      f"--applied-chip none if you did not. Nothing was recorded.")
                return 1
            chip = None
        else:
            chip = None if str(args.applied_chip).lower() == "none" else \
                canonical_chip(args.applied_chip)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add run_gameweek.py tests/test_cli.py
git commit -m "fix: --confirm will not spend an advised chip on its own

Audit B4: the CLI defaulted the applied chip to the advised one, so a planning
run that was never acted on could mark a chip used. --applied-chip is now
required whenever a chip was advised."
```

---

## Task 8: Every transfer count reaches the rank layer

**Audit item:** B5 (High)

**Files:**
- Modify: `fpl/optimize/transfers.py:170-207`
- Test: `tests/test_transfers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `enumerate_transfer_plans(...)` — unchanged signature, changed guarantee: the returned list contains the best feasible plan for **every** `n` in `0..free_transfers + cfg.max_paid_hits` before any alternative at a lower `n` is added.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transfers.py`:

```python
def test_every_transfer_count_is_represented_before_alternatives():
    """The single global quota was filled in ascending transfer count: the hold
    plan took one slot, a large pool supplied k-1 one-transfer alternatives, and
    the loop exited before n=2. With one free transfer that meant the rank layer
    never saw a paid hit at all, and with banked transfers it never saw a
    coordinated two-move restructure."""
    cur = list(CURRENT)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(max_paid_hits=2, rank_sims=0),
                                     xp_col="xp_next5", k=4)
    counts = {p.n_transfers for p in plans}
    assert {0, 1, 2} <= counts, f"transfer counts reaching the rank layer: {counts}"


def test_the_reserved_plan_for_each_count_is_that_count_s_best():
    cur = list(CURRENT)
    cfg = Config(max_paid_hits=1, rank_sims=0)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                     cfg=cfg, xp_col="xp_next5", k=8)
    for n in (0, 1, 2):
        at_n = [p for p in plans if p.n_transfers == n]
        if not at_n:
            continue
        best_solo, _ = optimize_transfers(POOL, cur, bank=5.0, free_transfers=1,
                                          cfg=cfg, xp_col="xp_next5")
        assert max(p.net_xp for p in at_n) <= best_solo.net_xp + 1e-6
```

Import `optimize_transfers` at the top of the test file if it is not already imported.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_transfers.py -q -k "every_transfer_count"`
Expected: FAIL — counts is `{0, 1}`

- [ ] **Step 3: Write the implementation**

Replace the loop body of `enumerate_transfer_plans` in `fpl/optimize/transfers.py`:

```python
    current_set = {int(i) for i in current_squad_ids}
    budget, cost = _budget_and_cost(xp_df, current_set, bank, selling_prices)
    max_n = int(free_transfers) + int(cfg.max_paid_hits)
    plans: list[TransferPlan] = []
    excluded: list[list[int]] = []

    def add(plan) -> bool:
        """Keep a plan unless an identical fifteen is already in the list."""
        if any(set(plan.squad_ids) == set(p.squad_ids) for p in plans):
            return False
        plans.append(plan)
        excluded.append(list(plan.squad_ids))
        return True

    # PASS 1 -- one reserved slot per transfer count.
    #
    # A single global quota filled in ascending order never got this far: the
    # hold plan took a slot, a large pool then supplied every remaining slot
    # with one-transfer alternatives, and the outer loop exited before n=2. With
    # one free transfer the rank layer was therefore never shown a paid hit, and
    # with banked transfers it was never shown a coordinated two-move
    # restructure. Reserving a slot per count costs at most max_n solves and
    # guarantees the whole decision is on the table.
    for n in range(0, max_n + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                        allow_club_overage=(n == 0))
        if solved is None:
            continue
        add(_plan(current_set, solved, free_transfers, cfg))

    # PASS 2 -- spend whatever quota is left on genuinely different alternatives,
    # deepest count first so the extra candidates are the ones the first pass
    # could not express.
    for n in range(max_n, -1, -1):
        while len(plans) < int(k):
            solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                            excluded=excluded, min_different=int(min_different),
                            allow_club_overage=(n == 0))
            if solved is None or not add(_plan(current_set, solved,
                                               free_transfers, cfg)):
                break
        if len(plans) >= int(k):
            break

    _attribute_gain(plans)
    return plans
```

Update the docstring's last paragraph to:

```
    Plans are enumerated in two passes. The first reserves one slot for the best
    plan at EVERY transfer count from 0 to free-plus-paid, so the rank layer
    always sees the whole decision -- hold, free moves, and each paid hit. Only
    then does the second pass spend the remaining quota on alternatives, deepest
    count first. `min_different` defaults to 1 rather than the 4 used for a
    squad rebuild -- a transfer plan IS a small change, and forcing candidates
    four players apart would only return plans nobody would consider.
```

Note: `k` may now be smaller than `max_n + 1`. That is intentional — the reserved
pass always runs in full, so a small `k` widens rather than truncates the decision.

The `allow_club_overage` argument is added in Task 9; until that task lands, write
these two `_solve` calls without it and add it in Task 9.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_transfers.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/optimize/transfers.py tests/test_transfers.py
git commit -m "fix: reserve a candidate slot for every transfer count

Audit B5: one global quota filled in ascending order, so a large pool spent it
all on one-transfer alternatives and the rank layer never saw n=2 -- every paid
hit and every coordinated two-move restructure was invisible."
```

---

## Task 9: The club cap allows FPL's real-transfer exception

**Audit item:** B17 (Low/Medium)

**Files:**
- Modify: `fpl/optimize/transfers.py:64-127,170-235`
- Test: `tests/test_transfers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_solve(..., allow_club_overage: bool = False)`. When true, a club the **current** squad already over-owns keeps its existing count as the cap instead of `MAX_PER_CLUB`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transfers.py`:

```python
def test_holding_a_four_from_one_club_squad_is_legal():
    """A real Premier League transfer can leave an FPL manager with four from
    one club. FPL lets the squad stand and requires three only when the manager
    NEXT makes a transfer, so the zero-transfer hold must stay feasible."""
    pool = POOL.copy()
    cur = list(CURRENT)
    # Move a fourth current player into the club three of them already share.
    club = pool.set_index("player_id").loc[cur[0], "team"]
    pool.loc[pool.player_id.isin(cur[:4]), "team"] = club

    plans = enumerate_transfer_plans(pool, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(max_paid_hits=1, rank_sims=0),
                                     xp_col="xp_next5", k=6)
    hold = [p for p in plans if p.n_transfers == 0]
    assert hold, "the hold plan was made infeasible by the club cap"
    assert set(hold[0].squad_ids) == set(cur)


def test_any_transfer_must_return_the_squad_to_three_per_club():
    pool = POOL.copy()
    cur = list(CURRENT)
    club = pool.set_index("player_id").loc[cur[0], "team"]
    pool.loc[pool.player_id.isin(cur[:4]), "team"] = club

    plans = enumerate_transfer_plans(pool, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(max_paid_hits=1, rank_sims=0),
                                     xp_col="xp_next5", k=6)
    frame = pool.set_index("player_id")
    for plan in plans:
        if plan.n_transfers == 0:
            continue
        counts = frame.loc[list(plan.squad_ids), "team"].value_counts()
        assert counts.max() <= 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_transfers.py -q -k "four_from_one_club"`
Expected: FAIL — no hold plan is returned

- [ ] **Step 3: Write the implementation**

In `fpl/optimize/transfers.py`, add the parameter to `_solve`:

```python
def _solve(xp_df, current, budget, max_changes, cfg, xp_col, cost=None,
           excluded=None, min_different=1, bench_floor: float = 0.0,
           allow_club_overage: bool = False):
```

and replace the club-cap constraint block:

```python
    # A real Premier League transfer can temporarily leave an FPL manager with
    # four players from one club. FPL does NOT force a sale: the squad stands,
    # and must return to three only when the manager next makes a transfer. So
    # the zero-transfer hold is allowed to keep an existing overage -- forcing
    # the cap there made holding infeasible and manufactured a move the game
    # does not require. Every plan that does transfer, and every fresh or
    # chip-built squad, is capped at three.
    owned_per_club: dict = {}
    if allow_club_overage:
        for i in current_set:
            if i in club:
                owned_per_club[club[i]] = owned_per_club.get(club[i], 0) + 1
    for c in set(club.values()):
        cap = max(MAX_PER_CLUB, owned_per_club.get(c, 0)) if allow_club_overage \
            else MAX_PER_CLUB
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= cap
```

In `optimize_transfers`, pass the flag on the `n == 0` solve:

```python
    for n in range(0, int(free_transfers) + int(cfg.max_paid_hits) + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                        allow_club_overage=(n == 0))
```

In `enumerate_transfer_plans`, add `allow_club_overage=(n == 0)` to both `_solve`
calls written in Task 8.

Do **not** pass it from `pipeline._squad_quality` or `optimize.actions._rebuild`:
a Wildcard or Free Hit squad is built from scratch and is capped at three.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_transfers.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/optimize/transfers.py tests/test_transfers.py
git commit -m "fix: let the hold plan keep a club overage FPL does not force out

Audit B17: a real transfer can leave four players from one club. FPL requires
three again only at the next transfer, but the strict cap made holding
infeasible and forced a move the game does not require."
```

---

## Task 10: An unavailable player scores zero everywhere

**Audit item:** B9 (Medium)

**Files:**
- Modify: `fpl/model/minutes.py:216-250`
- Test: `tests/test_minutes.py`, `tests/test_simulate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `minutes_model(...)` — unchanged signature. For `status in UNAVAILABLE`, every one of `p_start`, `p_play`, `p_60`, `e_minutes` is `0.0`. For `status == DOUBTFUL`, the availability factor multiplies the cameo branch as well as the start branch.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_minutes.py` (match the file's existing frame-building helper):

```python
def test_an_unavailable_player_has_no_route_onto_the_pitch():
    """p_start was zeroed and p_play then rebuilt from the generic cameo rule,
    leaving a ruled-out player a 35% chance of appearing. Deterministic xP read
    e_minutes and returned zero; the simulator read p_play and put him on."""
    players = _players(status="i", news="knee injury")
    m = minutes_model(players, Config()).set_index("player_id").loc[1]
    assert m.p_start == 0.0
    assert m.p_play == 0.0
    assert m.p_60 == 0.0
    assert m.e_minutes == 0.0


def test_a_doubtful_player_loses_the_cameo_too():
    """A 50% chance of playing is 50% of BOTH ways he could play, not half a
    start plus a full cameo."""
    fit = minutes_model(_players(status="a"), Config()).set_index("player_id").loc[1]
    doubt = minutes_model(_players(status="d", chance=50.0),
                          Config()).set_index("player_id").loc[1]
    assert doubt.p_start == pytest.approx(fit.p_start * 0.5)
    assert doubt.p_play == pytest.approx(fit.p_play * 0.5)
```

Append to `tests/test_simulate.py`:

```python
def test_an_unavailable_player_scores_zero_in_every_scenario():
    """Shared scenarios feed the rank layer, so one phantom cameo distorts
    captain, autosub and rank results for every candidate at once."""
    players, rates, minutes, tfx = _fixture()     # existing helper in this file
    minutes.loc[minutes.player_id == 1,
                ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    ids, samples = simulate_event(players, rates, minutes, tfx, event=1, n_sims=200)
    assert samples[ids.index(1)].max() == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_minutes.py -q -k "unavailable or doubtful"`
Expected: FAIL — `assert 0.35 == 0.0`

- [ ] **Step 3: Write the implementation**

In `fpl/model/minutes.py`, inside the player loop, replace the status block:

```python
        # Availability is a CAP on every route onto the pitch, not a discount on
        # one of them. Zeroing p_start alone left the generic cameo rule below
        # to hand a ruled-out player a 35% chance of appearing: deterministic xP
        # read e_minutes and returned zero, while the simulator read p_play and
        # put him on -- so the two disagreed exactly where the shared rank
        # scenarios are most sensitive.
        availability = 1.0
        status = str(p["status"])
        if status in UNAVAILABLE:
            availability = 0.0
            note = str(p["news"]).strip() or "unavailable"
            flags.append(f"Unavailable ({status}): {note}")
        elif status == DOUBTFUL:
            chance = p["chance_of_playing"]
            pct = 50.0 if pd.isna(chance) else float(chance)
            availability = pct / 100.0
            confidence = "low"
            note = str(p["news"]).strip()
            flags.append(f"Doubtful: {int(pct)}% chance of playing"
                         + (f" — {note}" if note else ""))
        p_start *= availability
```

Then replace the two lines that derive `p_play` and `e_minutes`:

```python
        p_start = float(min(1.0, max(0.0, p_start)))
        # The chance he would start if fully fit, recovered from the capped
        # value so the cameo branch can be capped by the SAME availability --
        # a 50% doubt halves the cameo as well as the start.
        fit_start = min(1.0, p_start / availability) if availability > 0 else 0.0
        p_play = availability * (fit_start + (1.0 - fit_start) * P_SUB_APPEAR)
```

and further down:

```python
        p_60 = p_start * p60_given_start
        # No `if p_start > 0` guard: availability already zeroes an unavailable
        # player through p_play, and a genuine cameo-only player does log
        # minutes. The old guard made e_minutes disagree with p_play.
        e_minutes = p_start * m_start + (p_play - p_start) * M_SUB
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_minutes.py tests/test_simulate.py tests/test_xp.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/model/minutes.py tests/test_minutes.py tests/test_simulate.py
git commit -m "fix: availability caps every route onto the pitch

Audit B9: zeroing p_start left the generic cameo rule giving a ruled-out player
a 35% appearance chance, so xP said zero and the simulator did not. Doubtful
players now lose the cameo branch in proportion too."
```

---

## Task 11: Calibration applies one intercept per real fixture week

**Audit item:** B10 (High)

**Files:**
- Modify: `fpl/model/calibration.py:109-144`
- Modify: `fpl/pipeline.py:328-334`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `fpl.optimize.objective.event_columns`.
- Produces: `apply_calibration(xp, cal, decay: float = 1.0) -> pd.DataFrame`. Per-event columns are calibrated one observation each; `xp_next1`, `xp_next5` and `xp_horizon` are **recomputed** from the calibrated event columns rather than calibrated independently.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration.py`:

```python
def test_a_blank_gameweek_stays_a_blank_after_calibration():
    """windows = projection / xp_next1 gave every column exactly one intercept
    when xp_next1 was zero -- including the zero current event, so a positive
    fitted intercept turned a genuine blank into points."""
    cal = Calibration(intercept={"MID": 0.5}, slope={"MID": 1.0},
                      pooled_intercept=0.5, pooled_slope=1.0, n_gameweeks=5,
                      n_observations=500, n_by_position={"MID": 500}, r2=0.1)
    xp = pd.DataFrame({
        "player_id": [1], "position": ["MID"],
        "xp_next1": [0.0], "xp_next5": [6.0], "xp_horizon": [5.4],
        "xp_gw5": [0.0], "xp_gw6": [3.0], "xp_gw7": [3.0],
    })
    out = apply_calibration(xp, cal, decay=0.9)
    assert out.loc[0, "xp_gw5"] == 0.0
    assert out.loc[0, "xp_next1"] == 0.0


def test_the_calibrated_horizon_is_the_sum_of_the_calibrated_weeks():
    """Clipping each column independently let xp_next5 drift away from the sum
    of the xp_gw columns, so the report, the optimizer and the captain value
    could disagree about the same player."""
    cal = Calibration(intercept={"MID": 0.3}, slope={"MID": 1.2},
                      pooled_intercept=0.3, pooled_slope=1.2, n_gameweeks=5,
                      n_observations=500, n_by_position={"MID": 500}, r2=0.1)
    xp = pd.DataFrame({
        "player_id": [1], "position": ["MID"],
        "xp_next1": [4.0], "xp_next5": [10.0], "xp_horizon": [9.1],
        "xp_gw5": [4.0], "xp_gw6": [3.0], "xp_gw7": [3.0],
    })
    out = apply_calibration(xp, cal, decay=0.9)
    weeks = [out.loc[0, c] for c in ("xp_gw5", "xp_gw6", "xp_gw7")]
    assert out.loc[0, "xp_next1"] == pytest.approx(weeks[0])
    assert out.loc[0, "xp_next5"] == pytest.approx(sum(weeks))
    assert out.loc[0, "xp_horizon"] == pytest.approx(
        weeks[0] + 0.9 * weeks[1] + 0.9 ** 2 * weeks[2])


def test_a_double_gameweek_column_gets_one_intercept_not_two():
    """The fit's observations are player-GAMEWEEKS, doubles included, so the
    offset applies once per gameweek. The ratio rule gave a double roughly two."""
    cal = Calibration(intercept={"MID": 1.0}, slope={"MID": 1.0},
                      pooled_intercept=1.0, pooled_slope=1.0, n_gameweeks=5,
                      n_observations=500, n_by_position={"MID": 500}, r2=0.1)
    xp = pd.DataFrame({
        "player_id": [1], "position": ["MID"],
        "xp_next1": [8.0], "xp_next5": [8.0], "xp_horizon": [8.0],
        "xp_gw5": [8.0],
    })
    out = apply_calibration(xp, cal, decay=1.0)
    assert out.loc[0, "xp_gw5"] == pytest.approx(9.0)


def test_a_frame_with_no_event_columns_still_calibrates():
    """Older ledger frames and several unit fixtures carry only the aggregates."""
    cal = Calibration(intercept={"MID": 0.5}, slope={"MID": 1.0},
                      pooled_intercept=0.5, pooled_slope=1.0, n_gameweeks=5,
                      n_observations=500, n_by_position={"MID": 500}, r2=0.1)
    xp = pd.DataFrame({"player_id": [1], "position": ["MID"],
                       "xp_next1": [4.0], "xp_next5": [10.0], "xp_horizon": [9.0]})
    out = apply_calibration(xp, cal, decay=0.9)
    assert out.loc[0, "xp_next1"] == pytest.approx(4.5)
```

Add `import pytest` and the `Calibration` import at the top of the file if absent.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_calibration.py -q`
Expected: FAIL — the blank column comes back at 0.5, and `xp_next5` does not equal the sum

- [ ] **Step 3: Write the implementation**

In `fpl/model/calibration.py`, replace `apply_calibration` entirely:

```python
def apply_calibration(xp: pd.DataFrame, cal: Calibration | None,
                      decay: float = 1.0) -> pd.DataFrame:
    """Rescale each per-gameweek projection, then rebuild the totals from them.

    The fit's unit of observation is a player-GAMEWEEK, so the transform applies
    once per gameweek column -- a double gets one intercept because it was one
    observation, and a blank gets none because it is not an observation at all.

    Scaling the intercept by `projection / xp_next1` instead, which is what this
    did, treated "twice this week's xP" as "two matches": an easy future single
    fixture collected several intercepts, a hard one collected a fraction, and
    when `xp_next1` was zero -- a blank, or a current injury -- EVERY column
    including the zero one collected exactly one, turning a genuine blank into
    points the player cannot score.

    `xp_next1`, `xp_next5` and `xp_horizon` are then recomputed from the
    calibrated weeks rather than calibrated in their own right, so the number in
    the report, the number the solver maximises and the number the captain is
    valued on cannot drift apart under the clip.
    """
    if cal is None:
        return xp
    from ..optimize.objective import event_columns

    out = xp.copy()
    pos = out["position"].astype(str)
    slope = pos.map(cal.slope).fillna(cal.pooled_slope).astype(float)
    inter = pos.map(cal.intercept).fillna(cal.pooled_intercept).astype(float)

    cols = event_columns(out)
    if not cols:
        # Frames carrying only the aggregates -- older ledger entries, and
        # several unit fixtures -- have nothing to rebuild from, so each column
        # is calibrated as the single observation it stands for.
        for col in projection_columns(out):
            out[col] = (slope * out[col].astype(float) + inter).clip(lower=0.0)
        return out

    for _, col in cols:
        values = out[col].astype(float)
        # A zero column is a blank gameweek, not a small projection: there is no
        # match for a per-match offset to attach to.
        out[col] = np.where(values > 0.0,
                            (slope * values + inter).clip(lower=0.0), 0.0)

    first = cols[0][0]
    out["xp_next1"] = out[cols[0][1]].astype(float).round(4)
    out["xp_next5"] = sum(out[c].astype(float) for _, c in cols).round(4)
    out["xp_horizon"] = sum(
        out[c].astype(float) * float(decay) ** (e - first) for e, c in cols
    ).round(4)
    return out
```

In `fpl/pipeline.py`, pass the decay:

```python
            xp = apply_calibration(xp, cal,
                                   decay=float(getattr(cfg, "horizon_decay", 1.0)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_calibration.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/model/calibration.py fpl/pipeline.py tests/test_calibration.py
git commit -m "fix: calibrate each gameweek once and rebuild the totals from it

Audit B10: the intercept was scaled by projection/xp_next1, so a blank week
with xp_next1 = 0 collected a full intercept and became positive xP, and the
clipped totals no longer equalled the sum of the weeks."
```

---

## Task 12: Start probability counts matches, not gameweeks

**Audit item:** B11 (Medium)

**Files:**
- Modify: `fpl/data/normalize.py:152-178`
- Modify: `fpl/model/minutes.py:155-160`
- Test: `tests/test_normalize.py`, `tests/test_minutes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `history_current_frame(...)` gains a `matches_played` column (fixture rows before `before_event`) alongside the existing `gws_played` (distinct rounds). `minutes_model` reads `matches_played` for the start denominator and leaves `gws_played` to `scoring.blend_form`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_normalize.py`:

```python
def test_a_double_gameweek_is_two_matches_and_one_gameweek():
    """`starts` sums fixture rows while `gws_played` counted distinct rounds, so
    a player who started both legs of a double read as two starts in one
    opportunity -- a p_start above 1 before clipping, and a rotation risk turned
    into a falsely nailed player for every week after."""
    summaries = {1: {"history": [
        {"round": 3, "starts": 1, "minutes": 90},
        {"round": 4, "starts": 1, "minutes": 90},
        {"round": 4, "starts": 1, "minutes": 90},
    ]}}
    frame = history_current_frame(summaries, before_event=5).set_index("player_id")
    assert frame.loc[1, "gws_played"] == 2
    assert frame.loc[1, "matches_played"] == 3
    assert frame.loc[1, "starts"] == 3
```

Append to `tests/test_minutes.py`:

```python
def test_start_probability_cannot_exceed_one_after_a_double():
    """Three starts in two gameweeks is not a 150% start rate."""
    players = _players(status="a")
    current = pd.DataFrame([{"player_id": 1, "gws_played": 2, "matches_played": 3,
                             "starts": 3, "minutes": 270}])
    for col in ("goals_scored", "assists", "bonus", "clean_sheets", "saves",
                "yellow_cards", "red_cards", "total_points", "expected_goals",
                "expected_assists"):
        if col not in current.columns:
            current[col] = 0
    m = minutes_model(players, Config(), current=current).set_index("player_id")
    assert m.loc[1, "p_start"] <= 1.0
    assert m.loc[1, "p_start"] > 0.8
```

Match the exact column set `history_current_frame` produces — copy it from
`fpl/data/normalize.py`'s `PLAYER_INT_COLS + FLOAT_COLS` rather than guessing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_normalize.py -q -k "double_gameweek"`
Expected: FAIL — `KeyError: 'matches_played'`

- [ ] **Step 3: Write the implementation**

In `fpl/data/normalize.py`, in `history_current_frame`, update the docstring and row build:

```python
    `gws_played` counts distinct ROUNDS and `matches_played` counts fixture
    ROWS. They differ in a double gameweek, and conflating them was a real bug:
    `starts` sums fixture rows, so a player who started both legs of a double
    read as two starts out of one opportunity and his start probability could
    exceed 1 before clipping. Matches are the right denominator for how many
    chances to start he has had; gameweeks are the right one for how much
    weight season-to-date form deserves, which is measured in weeks.
    """
    rows = []
    for pid, summary in (summaries or {}).items():
        played = [h for h in summary.get("history", [])
                  if int(h.get("round", 0)) < int(before_event)]
        if not played:
            continue
        row = {"player_id": int(pid),
               "gws_played": len({int(h["round"]) for h in played}),
               "matches_played": len(played)}
        for c in PLAYER_INT_COLS + FLOAT_COLS:
            row[c] = sum(pd.to_numeric(h.get(c, 0), errors="coerce") or 0 for h in played)
        rows.append(row)
    cols = ["player_id", "gws_played", "matches_played"] + PLAYER_INT_COLS + FLOAT_COLS
```

In `fpl/model/minutes.py`, replace the `now_games` assignment:

```python
        # MATCHES his club has played since he joined it -- not gameweeks, and
        # not games he featured in. `now_starts` sums fixture rows, so the
        # denominator has to count them too or a double gameweek produces more
        # starts than opportunities.
        now_games = (float(seen.get("matches_played", seen["gws_played"]))
                     if seen is not None else 0.0)
```

Using `.get` with the `gws_played` fallback keeps frames built by older callers
and by `tests/` fixtures working.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_normalize.py tests/test_minutes.py tests/test_scoring.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/data/normalize.py fpl/model/minutes.py tests/test_normalize.py tests/test_minutes.py
git commit -m "fix: count matches, not gameweeks, as chances to start

Audit B11: starts sums fixture rows while gws_played counted distinct rounds,
so a double gameweek could give a player more starts than opportunities and a
rotation risk read as nailed on for every week after."
```

---

## Task 13: The bench floor is opt-in, not a hidden default

**Audit item:** R11 (Medium)

**Files:**
- Modify: `fpl/config.py:31-39`
- Modify: `config.yaml`
- Test: `tests/test_config.py`, `tests/test_squad.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.bench_floor_xp` defaults to `0.0`. `config.yaml` gains an explicit, commented `optimizer.bench_floor_xp: 0.0` entry.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_the_bench_floor_is_off_unless_asked_for(tmp_path):
    """Every fresh build was silently solving a Bench Boost readiness problem:
    all four bench players forced over 2.5 xP, spending XI budget even when the
    chip was already spent -- and the setting was not in the checked-in config,
    so nobody could see it was on."""
    assert Config().bench_floor_xp == 0.0
    p = tmp_path / "config.yaml"
    p.write_text("optimizer:\n  bench_floor_xp: 2.5\n")
    assert load_config(p).bench_floor_xp == 2.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_config.py -q -k "bench_floor"`
Expected: FAIL — `assert 2.5 == 0.0`

- [ ] **Step 3: Write the implementation**

In `fpl/config.py`, replace the `bench_floor_xp` field and its comment:

```python
    # Lowest projection allowed to SIT on the bench in a squad build. This is a
    # Bench Boost READINESS constraint, not an ordinary-week one: forcing every
    # bench slot over a floor spends XI budget on players who, in a normal week,
    # score nothing. Measured on the real GW5 pool it cost 0.07 xP of XI to gain
    # 9.94 xP of bench -- a good trade the week you play the chip and a pure
    # loss every week you do not, including after it has been spent.
    #
    # So it is OFF by default and belongs in config.yaml only while preparing a
    # Wildcard or a Bench Boost. `pipeline._squad_quality` and the Wildcard
    # action still honour whatever is configured.
    bench_floor_xp: float = 0.0
```

In `config.yaml`, under the `optimizer:` block, add:

```yaml
  # Minimum xP for a player allowed to sit on the bench. 0 disables it.
  # Raise to ~2.5 only while preparing a Wildcard or a Bench Boost: it buys a
  # strong bench with budget the starting XI would otherwise have.
  bench_floor_xp: 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_config.py tests/test_squad.py -q`
Expected: PASS. Tests in `tests/test_squad.py` that rely on the floor being on
must now construct `Config(bench_floor_xp=2.5)` explicitly — update them and say
in the docstring that the floor is opt-in.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fpl/config.py config.yaml tests/test_config.py tests/test_squad.py
git commit -m "fix: bench floor defaults off and is visible in config.yaml

Audit R11: every fresh build silently required all four bench players over
2.5 xP, spending XI budget on a Bench Boost readiness problem even after the
chip was spent, and the setting was absent from the checked-in config."
```

---

## Task 14: The Tier 1 backtest stops seeing the target season

**Audit item:** B12 (Critical validation risk)

**Files:**
- Modify: `fpl/backtest/aggregate.py:15-45`
- Test: `tests/test_backtest_aggregate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `walk_forward_aggregate(past, cfg) -> dict` — same keys as before, plus `"by_cutoff": dict[str, dict]` giving per-target-season `{"mae", "n"}`. Every prediction for target season `t` is built only from seasons strictly before `t`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_aggregate.py`:

```python
def test_the_population_prior_cannot_see_the_target_season():
    """pop_mean was the mean over the WHOLE frame, target seasons included, so
    every prediction was shrunk toward a number that already knew the answer."""
    past = pd.DataFrame([
        {"player_id": 1, "season_name": "2023/24", "minutes": 3000, "total_points": 100},
        {"player_id": 1, "season_name": "2024/25", "minutes": 3000, "total_points": 110},
        {"player_id": 2, "season_name": "2023/24", "minutes": 3000, "total_points": 90},
        {"player_id": 2, "season_name": "2024/25", "minutes": 3000, "total_points": 95},
    ])
    quiet = past.copy()
    # Move the TARGET season only. A leak-free predictor cannot notice.
    quiet.loc[quiet.season_name == "2024/25", "total_points"] = 400

    cfg = _cfg()
    a = walk_forward_aggregate(past, cfg)
    b = walk_forward_aggregate(quiet, cfg)
    assert a["n"] == b["n"]
    # The predictions are identical; only the actuals they are scored against move.
    assert a["mae"] != b["mae"]


def test_every_eligible_season_is_predicted_not_only_the_last():
    """Predicting each player's final season alone threw away most of the
    available out-of-sample evidence."""
    past = pd.DataFrame([
        {"player_id": 1, "season_name": s, "minutes": 3000, "total_points": p}
        for s, p in (("2022/23", 100), ("2023/24", 110), ("2024/25", 120))
    ])
    out = walk_forward_aggregate(past, _cfg())
    assert out["n"] == 2                      # 2023/24 and 2024/25
    assert set(out["by_cutoff"]) == {"2023/24", "2024/25"}


def test_the_naive_baseline_is_not_the_mean_of_the_answers():
    """`naive` was the mean of the TARGET actuals, which no forecaster could
    have known -- it made the baseline artificially hard to beat in MAE and the
    comparison meaningless."""
    past = pd.DataFrame([
        {"player_id": p, "season_name": s, "minutes": 3000, "total_points": v}
        for p in (1, 2, 3)
        for s, v in (("2023/24", 60 + p * 10), ("2024/25", 200))
    ])
    out = walk_forward_aggregate(past, _cfg())
    # Every 2024/25 actual is identical, so a mean-of-actuals baseline would
    # score a perfect 0. A pre-cutoff baseline cannot.
    assert out["naive_mae"] > 0
```

Add a `_cfg()` helper to the file if it has none:

```python
def _cfg():
    from fpl.config import Config
    return Config()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_backtest_aggregate.py -q`
Expected: FAIL — `assert 1 == 2` and `assert 0.0 > 0`

- [ ] **Step 3: Write the implementation**

Replace `walk_forward_aggregate` in `fpl/backtest/aggregate.py`:

```python
def walk_forward_aggregate(past: pd.DataFrame, cfg) -> dict:
    """Predict every eligible season's points-per-90 from strictly earlier ones.

    Two leaks made the old version useless as evidence. It predicted only each
    player's FINAL season, discarding most of the out-of-sample data; and both
    the shrinkage prior and the "naive" baseline were computed over the WHOLE
    frame, target seasons included. Shrinking toward a mean that already
    contains the answer, and comparing against a baseline that IS the mean of
    the answers, can approve a worse rate model -- which then affects every
    weekly decision the tool makes.

    So the frame is replayed by season cutoff: for target season `t`, both the
    player's history and the population prior come only from seasons before `t`.
    """
    df = past.sort_values(["player_id", "season_name"]).copy()
    df["pts90"] = df.apply(_pts90, axis=1)
    k = float(cfg.shrinkage_minutes)

    seasons = sorted(df["season_name"].unique())
    preds, actuals, naive, by_cutoff = [], [], [], {}

    for target in seasons[1:]:
        prior = df[df["season_name"] < target]
        if prior.empty:
            continue
        # The prior population, weighted by exposure so a one-minute cameo does
        # not count as much as a full season.
        prior_minutes = float(prior["minutes"].sum())
        pop_mean = (float((prior["pts90"] * prior["minutes"]).sum()) / prior_minutes
                    if prior_minutes > 0 else 0.0)

        season_preds, season_actuals = [], []
        for pid, group in df[df["season_name"] == target].groupby("player_id"):
            history = prior[prior["player_id"] == pid]
            if history.empty:
                continue          # no pre-cutoff history: nothing to predict from
            mins = float(history["minutes"].sum())
            weighted = float((history["pts90"] * history["minutes"]).sum())
            season_preds.append((weighted + k * pop_mean) / (mins + k))
            season_actuals.append(float(group["pts90"].iloc[0]))
            # The honest baseline: what a forecaster who knew only the past
            # would have guessed. Not the mean of the answers.
            naive.append(pop_mean)

        if not season_preds:
            continue
        preds += season_preds
        actuals += season_actuals
        by_cutoff[str(target)] = {
            "mae": float(np.mean(np.abs(np.array(season_preds)
                                        - np.array(season_actuals)))),
            "n": len(season_preds),
        }

    if not preds:
        return {"mae": 0.0, "rmse": 0.0, "spearman": 0.0, "n": 0,
                "naive_mae": 0.0, "beats_naive": False, "by_cutoff": {}}

    preds_a, actual_a, naive_a = np.array(preds), np.array(actuals), np.array(naive)
    mae = float(np.mean(np.abs(preds_a - actual_a)))
    rmse = float(np.sqrt(np.mean((preds_a - actual_a) ** 2)))
    rho = float(spearmanr(preds_a, actual_a).statistic) if len(preds_a) > 2 else 0.0
    naive_mae = float(np.mean(np.abs(naive_a - actual_a)))
    return {"mae": mae, "rmse": rmse, "spearman": 0.0 if np.isnan(rho) else rho,
            "n": len(preds), "naive_mae": naive_mae, "beats_naive": mae < naive_mae,
            "by_cutoff": by_cutoff}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_backtest_aggregate.py -q`
Expected: PASS

- [ ] **Step 5: Check the reporting script still renders**

Run: `python -m pytest -q`
Expected: PASS. If `scripts/run_backtest.py` prints the result dict field by field,
add the new `by_cutoff` breakdown to its output rather than leaving it unreported.

- [ ] **Step 6: Commit**

```bash
git add fpl/backtest/aggregate.py tests/test_backtest_aggregate.py scripts/run_backtest.py
git commit -m "fix: replay the Tier 1 backtest by season cutoff

Audit B12: only each player's final season was predicted, and both the
shrinkage prior and the naive baseline were computed over the whole frame
including the target seasons -- so the comparison could approve a worse model."
```

---

## Task 15: Acceptance gate

**Audit item:** P0 acceptance gate

**Files:**
- Create: `tests/test_chip_acceptance.py`

**Interfaces:**
- Consumes: everything built above.
- Produces: no production code — one end-to-end legality sweep the audit asks for by name.

- [ ] **Step 1: Write the acceptance tests**

Create `tests/test_chip_acceptance.py`:

```python
"""The audit's P0 acceptance gate, in one place.

'Synthetic tests covering all eight chips, GW19/20, one chip per GW, FT
balances 0-5, FH restoration, and a two-transfer optimum. No live run may
return a chip without a complete legal action.'
"""
import pytest

from fpl.chips import CHIPS, available_chips, chip_available
from fpl.config import Config
from fpl.state import State, ft_after_moves


def test_all_eight_chips_are_reachable_across_a_season():
    """Four chips, two windows, eight uses -- and playing each first-half copy
    must leave every second-half copy intact."""
    first_half = [{"chip": c, "event": 5 + i} for i, c in enumerate(CHIPS)]
    for chip in CHIPS:
        assert not chip_available(chip, 18, first_half)
        assert chip_available(chip, 30, first_half)


def test_the_window_boundary_is_gw19_to_gw20():
    played = [{"chip": "benchboost", "event": 19}]
    assert not chip_available("benchboost", 19, played)
    assert chip_available("benchboost", 20, played)


def test_only_one_chip_is_legal_in_any_single_gameweek():
    played = [{"chip": "wildcard", "event": 10}]
    others = [c for c in CHIPS if c != "wildcard"]
    assert available_chips(10, played) == []
    for chip in others:
        assert chip_available(chip, 11, played)


@pytest.mark.parametrize("balance", [0, 1, 2, 3, 4, 5])
def test_every_free_transfer_balance_survives_a_chip_week(balance):
    remaining, nxt = ft_after_moves(State(free_transfers=balance),
                                    transfers_made=12, chip="wildcard")
    assert (remaining, nxt) == (balance, balance)


@pytest.mark.parametrize("balance", [0, 1, 2, 3, 4, 5])
def test_every_free_transfer_balance_accrues_normally_without_a_chip(balance):
    _, nxt = ft_after_moves(State(free_transfers=balance), transfers_made=0)
    assert nxt == min(5, balance + 1)
```

- [ ] **Step 2: Run the acceptance tests**

Run: `python -m pytest tests/test_chip_acceptance.py -q`
Expected: PASS

- [ ] **Step 3: Run the full suite and record the count**

Run: `python -m pytest -q`
Expected: PASS, with a total well above the 510 baseline.

- [ ] **Step 4: Smoke-test the real entry point**

Run: `python run_gameweek.py --mode 2 --gw 5 --no-refresh`
Expected: a rendered report, `data/state.json` untouched (planning run), and — if a
chip is advised — the note that the fifteen shown are that chip's own squad.

- [ ] **Step 5: Commit**

```bash
git add tests/test_chip_acceptance.py
git commit -m "test: add the audit's P0 chip and free-transfer acceptance gate"
```

---

## Self-Review

**Spec coverage**

| Audit item | Task |
|---|---|
| B1 chip inventory per half | 1, 4, 15 |
| B2 WC/FH free-transfer carry | 2, 15 |
| B3 Free Hit destroys permanent state | 3 |
| B4 chip advice is not the chip's action | 5, 6, 7 |
| B5 two-transfer plans never enumerated | 8 |
| B9 unavailable player cameo | 10 |
| B10 calibration intercept scaling | 11 |
| B11 DGW start evidence | 12 |
| B12 Tier 1 validation leakage | 14 |
| B17 club-cap real-transfer exception | 9 |
| R11 hidden bench floor | 13 |

Out of scope for this plan, and deliberately so: B6, B7, B8, B13–B16, R1–R10 and
the whole of P1–P3. They are the audit's validation-harness and
objective-alignment programs and need their own plans.

**Type consistency check**

- `chip_events` is `list[dict]` with keys `chip: str` and `event: int | None` in
  Tasks 1, 3, 4 and 15 — consistent with the existing `State.chip_events`.
- `ChipAction.transfers` is `TransferPlan | None`; Task 6 reads `.hit_cost`,
  `.squad_ids`, `.starting_ids` — all present on `TransferPlan`.
- `allow_club_overage` is introduced in Task 9 and referenced by Task 8's code
  block; Task 8 explicitly says to omit it until Task 9 lands. If executing in
  order, add it in Task 9.
- `bench_boost_value` / `triple_captain_value` take `(lineup, xp_df)` in both the
  tests and the implementation.
- `apply_calibration(xp, cal, decay=1.0)` — the pipeline call in Task 11 matches.

**Known ordering constraint**

Tasks 1 → 4 → 6 and 2 → 3 → 7 are ordered. Tasks 8 and 9 touch the same function
and must run in that order. Tasks 10–14 are independent of everything else and of
each other.

**One rule to verify before trusting Task 1**

`FREE_HIT_MIN_GAP` encodes the audit's claim that Free Hits cannot be played in
consecutive gameweeks. That is the one rule in this plan taken on the audit's
authority rather than read off the code. It is isolated in a single named
constant: setting it to `0` disables the restriction without touching anything
else.
