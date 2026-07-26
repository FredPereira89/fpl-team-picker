import json
import pytest
from fpl.cli import resolve_current_squad, record_transfers
from fpl.config import Config
from fpl.state import load_state, State


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
    assert live.free_transfers == 2  # unused FT rolls 1 -> 2
    assert live.warnings == []


def test_warns_when_tracked_state_drifts_from_history(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"free_transfers": 5, "last_event": 1, "chips_used": []}))
    client = FakeClient(picks=PICKS, history=HISTORY_NO_TRANSFERS)
    live, errors = resolve_current_squad(Config(entry_id=1), gw=2,
                                          state_path=state_path, client=client)
    assert errors == []
    assert live.free_transfers == 2  # trusts the derived value, not the stale 5
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
