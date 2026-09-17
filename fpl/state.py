"""Persisted free-transfer balance and chip usage.

The FT count is not available from any public endpoint - the only source is
auth-gated and off-limits - so it is tracked locally and reconciled against
entry/{id}/history/ event_transfers.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

FT_CAP = 5
CHIPS_PRESERVING_FT = {"wildcard", "freehit"}
# entry/{id}/history/ names two chips differently from the rest of this
# codebase. Reconciling without this mapping silently drops a Bench Boost or a
# Triple Captain from the used set, and the advisor offers a spent chip again.
API_CHIP_NAMES = {
    "wildcard": "wildcard",
    "freehit": "freehit",
    "bboost": "benchboost",
    "3xc": "triplecaptain",
    "manager": "manager",
}


def canonical_chip(name: str | None) -> str | None:
    """This codebase's name for a chip, whatever the API called it."""
    if not name:
        return None
    key = str(name).strip().lower()
    return API_CHIP_NAMES.get(key, key)


@dataclass
class State:
    free_transfers: int = 1
    last_event: int = 0
    chips_used: list[str] = field(default_factory=list)
    # The same chips, each with the gameweek it was played in. A bare name list
    # cannot say which half of the season a Wildcard belongs to, and -- the
    # failure that actually bit -- cannot tell the free-transfer reconciliation
    # which gameweek to treat as balance-preserving.
    chip_events: list[dict] = field(default_factory=list)
    # What was PAID for each owned player. FPL sells at purchase price plus half
    # the rise, and no public endpoint reports the purchase price, so the only
    # way to budget a transfer honestly is to remember it here.
    purchase_prices: dict[int, float] = field(default_factory=dict)
    # A squad confirmed by hand, and the gameweek it is the squad FOR.
    # entry/{id}/event/{gw}/picks exists only once GW{gw} has started, so
    # throughout the window when planning happens it can only report LAST
    # week's team: a transfer made before the deadline stays invisible until
    # after it. Recording the real 15 here is the only way the optimizer can
    # plan from what is actually owned rather than a week-old snapshot.
    squad: list[int] = field(default_factory=list)
    squad_event: int = 0
    bank: float = 0.0
    # Free transfers still unused *inside* last_event, as opposed to
    # `free_transfers`, which is the balance for the gameweek after it. Without
    # this, re-planning a gameweek whose moves were already confirmed read the
    # next week's balance -- the +1 had already accrued -- and the optimizer was
    # handed a free transfer that does not exist.
    free_transfers_remaining: int = 0


def _chip_records(raw: dict) -> list[dict]:
    """Chip records from stored JSON, back-filling files written before
    `chip_events` existed.

    Such a file still knows WHICH chips were used, just not when; recording
    those with a null event keeps the advisor correct across the upgrade
    rather than resurrecting spent chips.
    """
    records = raw.get("chip_events")
    if records:
        return [{"chip": canonical_chip(r.get("chip")),
                 "event": None if r.get("event") is None else int(r["event"])}
                for r in records if r.get("chip")]
    return [{"chip": canonical_chip(c), "event": None}
            for c in (raw.get("chips_used") or [])]


def load_state(path: Path, cfg) -> State:
    p = Path(path)
    if not p.exists():
        return State(free_transfers=cfg.free_transfers, last_event=0, chips_used=[])
    raw = json.loads(p.read_text())
    return State(
        free_transfers=int(raw.get("free_transfers", cfg.free_transfers)),
        last_event=int(raw.get("last_event", 0)),
        chips_used=list(raw.get("chips_used", [])),
        chip_events=_chip_records(raw),
        # JSON object keys are always strings; the rest of the codebase keys
        # players by int, so convert on the way back in.
        purchase_prices={int(k): float(v)
                         for k, v in (raw.get("purchase_prices") or {}).items()},
        squad=[int(i) for i in (raw.get("squad") or [])],
        squad_event=int(raw.get("squad_event", 0)),
        bank=float(raw.get("bank", 0.0)),
        free_transfers_remaining=int(raw.get("free_transfers_remaining", 0)),
    )


def save_state(state: State, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))


def record_chip(state: State, chip: str | None,
                event: int | None) -> tuple[list[str], list[dict]]:
    """Add one chip use to a state's chip history.

    A repeat of the same chip in the same gameweek is not a second use -- a
    confirmation re-run must be idempotent.
    """
    name = canonical_chip(chip)
    if name is None:
        return list(state.chips_used), [dict(r) for r in state.chip_events]
    events = [dict(r) for r in state.chip_events]
    if not any(r["chip"] == name and r.get("event") == event for r in events):
        events.append({"chip": name, "event": None if event is None else int(event)})
    names = list(state.chips_used)
    if name not in names:
        names.append(name)
    return names, events


def merge_chip_events(state: State, records: list[dict]) -> tuple[list[str], list[dict]]:
    """Fold API-derived chip records into local state.

    The API is authoritative about what was played and when; local state may
    additionally hold a chip confirmed for a gameweek FPL has not published
    yet, so this adds and never removes.
    """
    names, events = list(state.chips_used), [dict(r) for r in state.chip_events]
    for r in records or []:
        name = canonical_chip(r.get("chip"))
        if name is None:
            continue
        event = None if r.get("event") is None else int(r["event"])
        # A locally recorded use with no gameweek is the same use the API has
        # now dated. Upgrade it in place instead of double-counting it.
        undated = next((e for e in events
                        if e["chip"] == name and e.get("event") is None), None)
        if undated is not None and event is not None:
            undated["event"] = event
        elif not any(e["chip"] == name and e.get("event") == event for e in events):
            events.append({"chip": name, "event": event})
        if name not in names:
            names.append(name)
    return names, events


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


def advance_ft(state: State, transfers_made: int, chip: str | None = None) -> int:
    """Balance floors at 0 before the weekly +1 accrues. Chips preserve the balance."""
    return ft_after_moves(state, transfers_made, chip)[1]


def chips_from_history(entry_history: dict) -> list[dict]:
    """Chip records from entry/{id}/history/, oldest first.

    This is the only public source for which chips have actually been played:
    local state knows only what was confirmed through this tool, so a chip
    played in the FPL app is invisible without it.
    """
    out = []
    for c in (entry_history or {}).get("chips") or []:
        name = canonical_chip(c.get("name"))
        if name is None:
            continue
        event = c.get("event")
        out.append({"chip": name, "event": None if event is None else int(event)})
    return sorted(out, key=lambda r: (r["event"] is None, r["event"] or 0))


def reconcile(state: State, entry_history: dict) -> tuple[int, bool]:
    """Derive the FT balance from the API's per-event transfer counts.

    Seeds at 0, not 1: transfers before the GW1 deadline are unlimited, and FPL
    grants the first free transfer only *after* that deadline. Replaying GW1
    through advance_ft therefore turns 0 into the 1 you carry into GW2. Seeding
    at 1 would double-count that grant and hand the optimizer a second transfer
    it would actually pay 4 points for.

    A Wildcard or Free Hit gameweek preserves the balance however many
    transfers were made in it, so the replay reads the chip history too.
    Without that, a wildcard week's dozen transfers drained the derived balance
    to 1 -- and because a successful history fetch overrides local tracking,
    that wrong number replaced the correct one the following gameweek.
    """
    chip_by_event = {r["event"]: r["chip"] for r in chips_from_history(entry_history)
                     if r["event"] is not None}
    derived = State(free_transfers=0, last_event=0, chips_used=[])
    for event in (entry_history or {}).get("current", []):
        ev = int(event.get("event", derived.last_event))
        derived.free_transfers = advance_ft(
            derived, int(event.get("event_transfers", 0)), chip=chip_by_event.get(ev)
        )
        derived.last_event = ev
    matched = derived.free_transfers == state.free_transfers
    return derived.free_transfers, matched
