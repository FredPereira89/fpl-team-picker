"""Mode 2 CLI wiring: resolve a manager's live squad for weekly transfers.

Pure orchestration -- fetches entry_picks/entry_history via the given client
and free-transfer state via fpl.state, then hands (current_squad, bank,
free_transfers) to pipeline.run(mode=2, ...). Returns None (with a reason)
whenever a live squad can't be resolved, so the caller falls back to the
honest Mode 1 rebuild instead of guessing at a squad.
"""
from dataclasses import dataclass, field
from pathlib import Path

from .state import load_state, save_state, reconcile, advance_ft, State


@dataclass
class LiveSquad:
    current_squad: list[int]
    bank: float
    free_transfers: int
    warnings: list[str]
    purchase_prices: dict[int, float] = field(default_factory=dict)


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

    # A squad confirmed by hand for THIS gameweek beats the API, which cannot
    # see a transfer until after the deadline it was made for -- by which point
    # the advice is useless. Only for this gameweek: once GW{gw} has started,
    # picks becomes authoritative again and a stale override would be worse
    # than no override.
    if state.squad and int(state.squad_event) == int(gw):
        current_squad = list(state.squad)
        bank = float(state.bank)
        warnings.append(
            f"Using the confirmed squad recorded for GW{gw} rather than your "
            f"GW{prev_gw} picks — FPL does not publish a squad for a gameweek "
            f"that has not started."
        )
    confirmed = bool(state.squad) and int(state.squad_event) == int(gw)
    if confirmed:
        # reconcile() replays COMPLETED gameweeks, so transfers already made for
        # the gameweek being planned are not in it and the balance reads high.
        # The same hand-confirmation that fixed the squad fixes this.
        free_transfers = int(state.free_transfers)
    else:
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

    missing = [p for p in current_squad if p not in state.purchase_prices]
    if state.purchase_prices and missing:
        warnings.append(
            f"{len(missing)} of your 15 have no recorded purchase price -- those are "
            f"budgeted at market value, which overstates what they would sell for if "
            f"they have risen."
        )

    return LiveSquad(current_squad, bank, free_transfers, warnings,
                     state.purchase_prices), []


def record_transfers(state_path: Path, cfg, gw: int, transfers_made: int,
                     chip: str | None,
                     purchase_prices: dict[int, float] | None = None) -> None:
    state = load_state(state_path, cfg)
    new_ft = advance_ft(state, transfers_made, chip)
    chips_used = state.chips_used + ([chip] if chip and chip not in state.chips_used else [])
    prices = state.purchase_prices if purchase_prices is None else purchase_prices
    # A confirmed squad is a statement about what is owned, which a planning run
    # has no business overwriting -- it carries forward untouched.
    save_state(State(new_ft, gw, chips_used, dict(prices),
                     squad=list(state.squad), squad_event=int(state.squad_event),
                     bank=float(state.bank)), state_path)
