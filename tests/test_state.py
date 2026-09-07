import json
from pathlib import Path
from fpl.config import Config
from fpl.state import (
    State, load_state, save_state, advance_ft, reconcile, FT_CAP,
    chips_from_history, record_chip, merge_chip_events, ft_after_moves,
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


def test_save_then_load_roundtrips_a_confirmed_squad(tmp_path):
    """FPL exposes picks only for gameweeks that have started, so a transfer
    made during the planning window is invisible until after the deadline it
    was made for. A squad confirmed by hand has to survive in state."""
    p = tmp_path / "state.json"
    save_state(State(free_transfers=0, last_event=3, chips_used=[],
                     squad=[1, 2, 3], squad_event=4, bank=0.3), p)
    s = load_state(p, Config(free_transfers=1))
    assert s.squad == [1, 2, 3]
    assert s.squad_event == 4
    assert s.bank == 0.3


def test_state_without_a_recorded_squad_reads_as_empty(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"free_transfers": 1, "last_event": 2}))
    s = load_state(p, Config(free_transfers=1))
    assert s.squad == []
    assert s.squad_event == 0


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


# --- Chip-aware reconciliation (2026-09-07 review) ---

def test_reconcile_does_not_spend_free_transfers_on_a_wildcard_week():
    """A Wildcard makes every transfer free and leaves the balance untouched.
    Replaying the history without reading the chip list drained a banked 2 down
    to 1 -- and because the derived value overrides local tracking, that wrong
    number then replaced the correct one."""
    history = {
        "current": [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 12},
        ],
        "chips": [{"name": "wildcard", "time": "2026-08-30T10:00:00Z", "event": 3}],
    }
    ft, _ = reconcile(State(), history)
    assert ft == 3  # 2 banked going into GW3, untouched, +1 for GW4


def test_reconcile_does_not_spend_free_transfers_on_a_free_hit_week():
    history = {
        "current": [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 11},
        ],
        "chips": [{"name": "freehit", "event": 2}],
    }
    ft, _ = reconcile(State(), history)
    assert ft == 2


def test_reconcile_still_spends_transfers_in_a_bench_boost_week():
    """Bench Boost does not touch transfers, so its gameweek is an ordinary one."""
    history = {
        "current": [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 1},
        ],
        "chips": [{"name": "bboost", "event": 2}],
    }
    ft, _ = reconcile(State(), history)
    assert ft == 1


def test_chips_from_history_translates_the_api_chip_names():
    """entry/{id}/history/ calls them bboost and 3xc; everything else in this
    codebase says benchboost and triplecaptain. Without the translation a spent
    chip reads as a different, unspent one and the advisor offers it again."""
    got = chips_from_history({"chips": [
        {"name": "3xc", "event": 4},
        {"name": "bboost", "event": 2},
    ]})
    assert got == [{"chip": "benchboost", "event": 2},
                   {"chip": "triplecaptain", "event": 4}]


def test_chips_from_history_tolerates_an_entry_with_no_chips():
    assert chips_from_history({"current": []}) == []


def test_recording_a_chip_stores_the_gameweek_it_was_played_in():
    names, events = record_chip(State(), "wildcard", 8)
    assert names == ["wildcard"]
    assert events == [{"chip": "wildcard", "event": 8}]


def test_recording_the_same_chip_twice_in_one_gameweek_is_not_two_uses():
    """Confirming a gameweek twice is a re-run, not a second chip."""
    names, events = record_chip(State(), "wildcard", 8)
    again = State(chips_used=names, chip_events=events)
    names2, events2 = record_chip(again, "wildcard", 8)
    assert names2 == ["wildcard"]
    assert events2 == events


def test_a_second_wildcard_in_a_later_gameweek_is_a_separate_use():
    """FPL grants two Wildcards a season. A name-only record cannot tell them
    apart; the gameweek can."""
    names, events = record_chip(State(), "wildcard", 5)
    second = record_chip(State(chips_used=names, chip_events=events), "wildcard", 25)
    assert second[1] == [{"chip": "wildcard", "event": 5},
                         {"chip": "wildcard", "event": 25}]


def test_merging_api_chips_dates_a_locally_recorded_use():
    """Local state records a chip when it is confirmed, before FPL publishes
    it. Once the API dates it, that is the same use, not a second one."""
    state = State(chips_used=["freehit"], chip_events=[{"chip": "freehit", "event": None}])
    names, events = merge_chip_events(state, [{"chip": "freehit", "event": 6}])
    assert names == ["freehit"]
    assert events == [{"chip": "freehit", "event": 6}]


def test_merging_api_chips_adds_one_played_outside_this_tool():
    """A chip played in the FPL app is invisible to local state until this
    merge -- and the advisor would keep recommending it."""
    names, events = merge_chip_events(State(), [{"chip": "3xc", "event": 4}])
    assert names == ["triplecaptain"]
    assert events == [{"chip": "triplecaptain", "event": 4}]


def test_a_state_file_written_before_chip_events_keeps_its_chips(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"free_transfers": 1, "last_event": 5,
                             "chips_used": ["wildcard"]}))
    s = load_state(p, Config())
    assert s.chips_used == ["wildcard"]
    assert s.chip_events == [{"chip": "wildcard", "event": None}]


def test_ft_after_moves_separates_this_week_from_next():
    """The balance left inside a gameweek is one short of the balance carried
    into the next one -- conflating them handed a re-planned gameweek a free
    transfer that had already been spent."""
    remaining, nxt = ft_after_moves(State(free_transfers=2), transfers_made=1)
    assert (remaining, nxt) == (1, 2)


def test_ft_after_moves_leaves_a_wildcard_week_whole():
    remaining, nxt = ft_after_moves(State(free_transfers=2), transfers_made=9,
                                    chip="wildcard")
    assert (remaining, nxt) == (2, 3)

