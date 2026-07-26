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
    # Corrected fixture: 0 transfers in events 1-2 (FT rolls 1->2->3 across
    # the season), then 2 transfers in event 3 (1 free + 1 paid hit),
    # leaving exactly 2 for the next event. The brief's original fixture
    # used event_transfers=1 for event 3, which under the correct
    # advance_ft formula (min(5, max(0, FT - used) + 1), applied
    # iteratively across all 3 events starting from the universal FT=1
    # season seed) actually yields ft=3, not 2 -- a genuine arithmetic
    # bug in the original test data, verified by hand execution before
    # dispatch. This fixture is hand-verified to produce ft==2, matched==True.
    state = State(free_transfers=2, last_event=3, chips_used=[])
    history = {"current": [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 2},
    ]}
    ft, matched = reconcile(state, history)
    assert matched is True
    assert ft == 2


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
