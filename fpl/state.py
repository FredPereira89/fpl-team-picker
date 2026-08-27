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

    Seeds at 0, not 1: transfers before the GW1 deadline are unlimited, and FPL
    grants the first free transfer only *after* that deadline. Replaying GW1
    through advance_ft therefore turns 0 into the 1 you carry into GW2. Seeding
    at 1 would double-count that grant and hand the optimizer a second transfer
    it would actually pay 4 points for.
    """
    derived = State(free_transfers=0, last_event=0, chips_used=[])
    for event in entry_history.get("current", []):
        derived.free_transfers = advance_ft(derived, int(event.get("event_transfers", 0)))
        derived.last_event = int(event.get("event", derived.last_event))
    matched = derived.free_transfers == state.free_transfers
    return derived.free_transfers, matched
