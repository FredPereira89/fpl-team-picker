import json
import pytest
from fpl.cli import (resolve_current_squad, record_transfers,
                     carry_purchase_prices)
from fpl.config import Config
from fpl.state import load_state, save_state, State


class FakeClient:
    def __init__(self, picks=None, history=None, picks_error=None, history_error=None):
        self._picks = picks
        self._history = history
        self._picks_error = picks_error
        self._history_error = history_error

    def entry_picks(self, entry_id, gw):
        if self._picks_error:
            raise self._picks_error
        return self._picks

    def entry_history(self, entry_id):
        if self._history_error:
            raise self._history_error
        return self._history


PICKS = {
    "entry_history": {"bank": 5, "value": 1000, "event_transfers": 0},
    "picks": [{"element": i, "position": i, "multiplier": 1,
               "is_captain": i == 1, "is_vice_captain": i == 2} for i in range(1, 16)],
}
HISTORY_NO_TRANSFERS = {"current": [{"event": 1, "event_transfers": 0}]}


def test_returns_none_without_entry_id(tmp_path):
    live, errors = resolve_current_squad(Config(entry_id=None), gw=2,
                                          state_path=tmp_path / "state.json",
                                          client=FakeClient())
    assert live is None
    assert "entry_id" in errors[0]


def test_returns_none_for_gw1(tmp_path):
    live, errors = resolve_current_squad(Config(entry_id=1), gw=1,
                                          state_path=tmp_path / "state.json",
                                          client=FakeClient())
    assert live is None
    assert "previous gameweek" in errors[0]


def test_returns_none_when_picks_fetch_fails(tmp_path):
    client = FakeClient(picks_error=RuntimeError("HTTP 404"))
    live, errors = resolve_current_squad(Config(entry_id=1), gw=2,
                                          state_path=tmp_path / "state.json",
                                          client=client)
    assert live is None
    assert "GW1 picks" in errors[0]


def test_happy_path_resolves_squad_bank_and_free_transfers(tmp_path):
    client = FakeClient(picks=PICKS, history=HISTORY_NO_TRANSFERS)
    live, errors = resolve_current_squad(Config(entry_id=1, free_transfers=1), gw=2,
                                          state_path=tmp_path / "state.json",
                                          client=client)
    assert errors == []
    assert live.current_squad == list(range(1, 16))
    assert live.bank == 0.5
    assert live.free_transfers == 1  # GW1 banks nothing; GW2 gets the first FT
    assert live.warnings == []


def test_warns_when_tracked_state_drifts_from_history(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"free_transfers": 5, "last_event": 1, "chips_used": []}))
    client = FakeClient(picks=PICKS, history=HISTORY_NO_TRANSFERS)
    live, errors = resolve_current_squad(Config(entry_id=1), gw=2,
                                          state_path=state_path, client=client)
    assert errors == []
    assert live.free_transfers == 1  # trusts the derived value, not the stale 5
    assert live.warnings and "drifted" in live.warnings[0]


def test_falls_back_to_local_state_when_history_fetch_fails(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"free_transfers": 3, "last_event": 1, "chips_used": []}))
    client = FakeClient(picks=PICKS, history_error=RuntimeError("network down"))
    live, errors = resolve_current_squad(Config(entry_id=1), gw=2,
                                          state_path=state_path, client=client)
    assert errors == []
    assert live.free_transfers == 3
    assert live.warnings and "assuming" in live.warnings[0]


def test_record_transfers_advances_and_persists_state(tmp_path):
    state_path = tmp_path / "state.json"
    record_transfers(state_path, Config(free_transfers=1), gw=2, transfers_made=1, chip=None)
    s = load_state(state_path, Config())
    assert s.free_transfers == 1  # used the 1 FT, floors at 0, +1 accrues
    assert s.last_event == 2


def test_record_transfers_tracks_chip_usage(tmp_path):
    state_path = tmp_path / "state.json"
    record_transfers(state_path, Config(free_transfers=1), gw=2, transfers_made=15, chip="wildcard")
    s = load_state(state_path, Config())
    assert s.chips_used == ["wildcard"]
    assert s.free_transfers == 2  # wildcard preserves balance, still accrues


# --- P6: purchase prices (2026-08-27 audit) ---

def test_recording_a_gameweek_keeps_the_price_paid_for_each_player(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    record_transfers(path, cfg, gw=2, transfers_made=1, chip=None,
                     purchase_prices={101: 5.5, 303: 7.0})
    assert load_state(path, cfg).purchase_prices == {101: 5.5, 303: 7.0}


def test_resolved_squad_carries_the_recorded_purchase_prices(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(entry_id=42, free_transfers=1)
    save_state(State(free_transfers=1, last_event=1, chips_used=[],
                     purchase_prices={1: 4.5}), path)
    live, _ = resolve_current_squad(cfg, gw=2, state_path=path, client=FakeClient(picks=PICKS, history=HISTORY_NO_TRANSFERS))
    assert live.purchase_prices == {1: 4.5}


# --- the ledger must track the squad you own, not the one you were offered ---

def test_purchase_prices_track_the_squad_you_own_not_the_one_recommended():
    """A recommendation is a proposal, not a transaction.

    run_gameweek.py wrote `rec.squad_ids` -- the optimizer's output -- so any
    advice the user declined leaked into the price ledger as if they had acted
    on it. Real state.json after two gameweeks held a purchase price for Leno,
    who was only ever *suggested*, and none for Raya, who was actually owned.
    That corrupts the selling values P6 exists to get right.
    """
    owned = [1, 2, 3]
    recommended_instead = [1, 2, 99]
    paid = {1: 6.0, 2: 5.5}
    market = {1: 6.2, 2: 5.5, 3: 4.5, 99: 12.0}

    carried = carry_purchase_prices(owned, paid, market)

    assert set(carried) == set(owned)
    assert 99 not in carried, "a player who was only recommended was never bought"
    assert carried[1] == 6.0, "a held player keeps what he cost, not what he is worth now"
    assert carried[3] == 4.5, "an unrecorded player falls back to market value"
    assert recommended_instead  # documents the squad that must NOT be recorded

