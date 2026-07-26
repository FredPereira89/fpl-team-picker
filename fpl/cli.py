"""Mode 2 CLI wiring: resolve a manager's live squad for weekly transfers.

Pure orchestration -- fetches entry_picks/entry_history via the given client
and free-transfer state via fpl.state, then hands (current_squad, bank,
free_transfers) to pipeline.run(mode=2, ...). Returns None (with a reason)
whenever a live squad can't be resolved, so the caller falls back to the
honest Mode 1 rebuild instead of guessing at a squad.
"""
from dataclasses import dataclass
from pathlib import Path

from .state import load_state, save_state, reconcile, advance_ft, State


@dataclass
class LiveSquad:
    current_squad: list[int]
    bank: float
    free_transfers: int
    warnings: list[str]


def resolve_current_squad(cfg, gw: int, state_path: Path, client):
    if cfg.entry_id is None:
        return None, ["Mode 2 needs `entry_id` set in config.yaml."]

    prev_gw = gw - 1
    if prev_gw < 1:
        return None, [f"Mode 2 needs a completed previous gameweek (GW{gw} has none)."]

    try:
        picks = client.entry_picks(cfg.entry_id, prev_gw)
    except Exception as e:
        return None, [f"Could not fetch your GW{prev_gw} picks ({e})."]

    current_squad = [int(p["element"]) for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0

    state_existed = Path(state_path).exists()
    state = load_state(state_path, cfg)
    warnings: list[str] = []
    try:
        history = client.entry_history(cfg.entry_id)
        free_transfers, matched = reconcile(state, history)
        if not matched and state_existed:
            warnings.append(
                f"Free-transfer count drifted from tracked state ({state.free_transfers}) "
                f"-- using {free_transfers} derived from your FPL transfer history."
            )
    except Exception:
        free_transfers = state.free_transfers
        warnings.append(
            f"Could not verify free transfers against FPL history -- assuming "
            f"{free_transfers} from local tracking."
        )

    return LiveSquad(current_squad, bank, free_transfers, warnings), []


def record_transfers(state_path: Path, cfg, gw: int, transfers_made: int, chip: str | None) -> None:
    state = load_state(state_path, cfg)
    new_ft = advance_ft(state, transfers_made, chip)
    chips_used = state.chips_used + ([chip] if chip and chip not in state.chips_used else [])
    save_state(State(new_ft, gw, chips_used), state_path)
