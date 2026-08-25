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


@dataclass
class State:
    free_transfers: int = 1
    last_event: int = 0
    chips_used: list[str] = field(default_factory=list)


def load_state(path: Path, cfg) -> State:
    p = Path(path)
    if not p.exists():
        return State(free_transfers=cfg.free_transfers, last_event=0, chips_used=[])
    raw = json.loads(p.read_text())
    return State(
        free_transfers=int(raw.get("free_transfers", cfg.free_transfers)),
        last_event=int(raw.get("last_event", 0)),
        chips_used=list(raw.get("chips_used", [])),
    )


def save_state(state: State, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))


def advance_ft(state: State, transfers_made: int, chip: str | None = None) -> int:
    """Balance floors at 0 before the weekly +1 accrues. Chips preserve the balance."""
    used = 0 if chip in CHIPS_PRESERVING_FT else int(transfers_made)
    return min(FT_CAP, max(0, int(state.free_transfers) - used) + 1)


def reconcile(state: State, entry_history: dict) -> tuple[int, bool]:
    """Derive the FT balance from the API's per-event transfer counts.

    GW1 has no free-transfer concept -- squad changes before the season's
    first deadline are unlimited and free, and FT accrual only starts once
    that deadline has passed (confirmed against the official rules: "Once
    the first Gameweek deadline of the season has passed, managers are
    given ONE free transfer for each Gameweek"). The free_transfers=1
    baseline below already represents that starting balance for GW2, so
    event 1 must be skipped rather than folded into the roll-over -- treating
    it like a normal accruing week silently credits a phantom extra
    transfer every season.
    """
    derived = State(free_transfers=1, last_event=0, chips_used=[])
    for event in entry_history.get("current", []):
        eid = int(event.get("event", derived.last_event))
        if eid <= 1:
            derived.last_event = eid
            continue
        derived.free_transfers = advance_ft(derived, int(event.get("event_transfers", 0)))
        derived.last_event = eid
    matched = derived.free_transfers == state.free_transfers
    return derived.free_transfers, matched
