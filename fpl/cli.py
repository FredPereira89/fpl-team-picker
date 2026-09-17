"""Mode 2 CLI wiring: resolve a manager's live squad for weekly transfers.

Pure orchestration -- fetches entry_picks/entry_history via the given client
and free-transfer state via fpl.state, then hands (current_squad, bank,
free_transfers) to pipeline.run(mode=2, ...). Returns None (with a reason)
whenever a live squad can't be resolved, so the caller falls back to the
honest Mode 1 rebuild instead of guessing at a squad.
"""
from dataclasses import dataclass, field
from pathlib import Path

from .state import (load_state, save_state, reconcile, ft_after_moves, record_chip,
                    merge_chip_events, chips_from_history, canonical_chip, State)


@dataclass
class LiveSquad:
    current_squad: list[int]
    bank: float
    free_transfers: int
    warnings: list[str]
    purchase_prices: dict[int, float] = field(default_factory=dict)
    # Chips already spent, local records merged with the API's. The advisor
    # takes this; passing it an empty list -- which the pipeline did until this
    # was wired -- recommends a chip that is gone.
    chips_used: list[str] = field(default_factory=list)
    # The same, with the gameweek each was played in, ready to be written back
    # on confirmation so a chip played in the FPL app survives in local state.
    chip_events: list[dict] = field(default_factory=list)
    # True when the squad above was restored from base state because a Free
    # Hit expired, rather than read from the previous gameweek's picks.
    restored_from_freehit: bool = False
    # The entry's OWN opening gameweek. Neither a Wildcard nor a Free Hit may be
    # played in it, and an entry that joined in GW7 has a different one from an
    # entry that started the season.
    first_event: int = 1


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

    # The chip history is a fact about the season, not about which squad is
    # live, so it is read whether or not the squad was confirmed by hand.
    history = None
    try:
        history = client.entry_history(cfg.entry_id)
    except Exception:
        history = None

    if confirmed:
        # reconcile() replays COMPLETED gameweeks, so transfers already made for
        # the gameweek being planned are not in it and the balance reads high.
        # The same hand-confirmation that fixed the squad fixes this.
        free_transfers = int(state.free_transfers)
        if int(state.last_event) >= int(gw):
            # This gameweek's moves were already confirmed: `free_transfers` is
            # next week's balance, with the weekly +1 already added.
            free_transfers = int(state.free_transfers_remaining)
    elif history is not None:
        free_transfers, matched = reconcile(state, history)
        if not matched and state_existed:
            warnings.append(
                f"Free-transfer count drifted from tracked state ({state.free_transfers}) "
                f"-- using {free_transfers} derived from your FPL transfer history."
            )
    else:
        free_transfers = state.free_transfers
        warnings.append(
            f"Could not verify free transfers against FPL history -- assuming "
            f"{free_transfers} from local tracking."
        )

    # An entry that joined mid-season has its own opening gameweek, and neither
    # a Wildcard nor a Free Hit may be played in it.
    entry_events = [int(e["event"]) for e in (history or {}).get("current", [])
                    if e.get("event")]
    first_event = min(entry_events) if entry_events else 1

    chips_used, chip_events = merge_chip_events(state, chips_from_history(history or {}))
    new_to_state = [c for c in chips_used if c not in state.chips_used]
    if new_to_state:
        warnings.append(
            f"FPL reports {', '.join(new_to_state)} already played — those are "
            f"excluded from this week's chip advice."
        )

    missing = [p for p in current_squad if p not in purchase_prices]
    if purchase_prices and missing:
        warnings.append(
            f"{len(missing)} of your 15 have no recorded purchase price -- those are "
            f"budgeted at market value, which overstates what they would sell for if "
            f"they have risen."
        )

    return LiveSquad(current_squad, bank, free_transfers, warnings,
                     purchase_prices, chips_used, chip_events,
                     restored_from_freehit=restored,
                     first_event=first_event), []


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
    # A confirmed squad is a statement about what is owned. A planning run has
    # no business overwriting it, so it carries forward untouched unless this
    # confirmation names the squad that was actually applied.
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
