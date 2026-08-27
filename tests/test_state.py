import json
from pathlib import Path
from fpl.config import Config
from fpl.state import (
    State, load_state, save_state, advance_ft, reconcile, FT_CAP,
)


def test_load_state_seeds_from_config_when_file_absent(tmp_path):
    s = load_state(tmp_path / "state.json", Config(free_transfers=2))
    assert s.free_transfers == 2
    assert s.chips_used == []


def test_save_then_load_roundtrips(tmp_path):
    p = tmp_path / "state.json"
    save_state(State(free_transfers=3, last_event=5, chips_used=["wildcard"]), p)
    s = load_state(p, Config(free_transfers=1))
    assert s.free_transfers == 3
    assert s.last_event == 5
    assert s.chips_used == ["wildcard"]


def test_unused_transfer_accrues():
    assert advance_ft(State(1, 1, []), transfers_made=0) == 2


def test_ft_caps_at_five():
    assert advance_ft(State(FT_CAP, 1, []), transfers_made=0) == FT_CAP
    assert advance_ft(State(FT_CAP, 1, []), transfers_made=1) == FT_CAP


def test_using_the_free_transfer_returns_to_one():
    assert advance_ft(State(1, 1, []), transfers_made=1) == 1


def test_taking_hits_floors_the_balance_at_zero_before_accruing():
    # 2 FT, 3 transfers made (one -4 hit) -> balance 0, then +1
    assert advance_ft(State(2, 1, []), transfers_made=3) == 1


def test_wildcard_preserves_the_balance_and_still_accrues():
    assert advance_ft(State(3, 1, []), transfers_made=9, chip="wildcard") == 4


def test_free_hit_preserves_the_balance():
    assert advance_ft(State(2, 1, []), transfers_made=11, chip="freehit") == 3


def test_bench_boost_does_not_preserve_the_balance():
    assert advance_ft(State(2, 1, []), transfers_made=2, chip="benchboost") == 1


def test_reconcile_matches_when_history_agrees():
    # 0 transfers in GW1 (unlimited pre-deadline, so nothing is banked) leaves 1
    # for GW2; 0 used in GW2 banks a second for GW3; the 2 made in GW3 spend both
    # free transfers, so GW4 starts from 0 and accrues exactly 1. Tracked state
    # agreeing with that derivation is what `matched` reports.
    state = State(free_transfers=1, last_event=3, chips_used=[])
    history = {"current": [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 2},
    ]}
    ft, matched = reconcile(state, history)
    assert matched is True
    assert ft == 1


def test_reconcile_reports_drift_and_prefers_derived_value():
    state = State(free_transfers=5, last_event=3, chips_used=[])
    history = {"current": [
        {"event": 1, "event_transfers": 1},
        {"event": 2, "event_transfers": 1},
        {"event": 3, "event_transfers": 1},
    ]}
    ft, matched = reconcile(state, history)
    assert matched is False
    assert ft == 1


def test_reconcile_gives_one_free_transfer_for_gw2():
    """GW1 transfers are unlimited, so GW2 starts with exactly 1 FT -- never 2.

    FPL grants the first free transfer *after* the GW1 deadline ("1 base +
    max_extra_free_transfers=4" in bootstrap game_settings). Seeding the
    pre-GW1 balance at 1 double-counts that grant and makes the optimizer
    spend a transfer it has to pay 4 points for.
    """
    state = State(free_transfers=1, last_event=1, chips_used=[])
    history = {"current": [{"event": 1, "event_transfers": 0}]}
    ft, _ = reconcile(state, history)
    assert ft == 1


def test_reconcile_banks_an_extra_transfer_after_an_unused_gw2():
    """One unused FT in GW2 rolls into GW3 as 2."""
    history = {"current": [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
    ]}
    ft, _ = reconcile(State(), history)
    assert ft == 2


def test_reconcile_spends_the_free_transfer_in_gw2():
    """Using the single GW2 transfer leaves 1 for GW3, not 0."""
    history = {"current": [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 1},
    ]}
    ft, _ = reconcile(State(), history)
    assert ft == 1


# --- P6: purchase prices (2026-08-27 audit) ---

def test_purchase_prices_survive_a_save_load_cycle(tmp_path):
    """Selling value needs the price PAID, and no public endpoint reports it —
    if state.json loses it, the transfer budget silently reverts to market value.
    """
    path = tmp_path / "state.json"
    save_state(State(free_transfers=2, last_event=3, chips_used=[],
                     purchase_prices={101: 5.5, 202: 12.0}), path)
    back = load_state(path, Config())
    assert back.purchase_prices == {101: 5.5, 202: 12.0}


def test_purchase_prices_default_to_empty_for_a_squad_never_recorded(tmp_path):
    assert load_state(tmp_path / "missing.json", Config()).purchase_prices == {}
