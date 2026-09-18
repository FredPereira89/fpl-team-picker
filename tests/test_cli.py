import json
import pytest
from fpl.cli import resolve_current_squad, record_transfers
from fpl.config import Config
from fpl.state import load_state, save_state, State
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[1]


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


def test_confirmed_squad_for_this_gameweek_overrides_the_api_picks(tmp_path):
    """entry/{id}/event/{gw}/picks only exists once GW{gw} has started, so during
    the window when planning actually happens it can only return LAST week's
    squad. A transfer made before the deadline is therefore invisible, and the
    optimiser would keep recommending players you already bought."""
    p = tmp_path / "state.json"
    save_state(State(free_transfers=0, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), p)
    live, errors = resolve_current_squad(Config(entry_id=1), gw=4, state_path=p,
                                         client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert errors == []
    assert live.current_squad == list(range(100, 115))
    assert live.bank == 0.0
    assert any("confirmed squad" in w for w in live.warnings)


def test_confirmed_squad_also_fixes_the_free_transfer_count(tmp_path):
    """reconcile() derives free transfers from COMPLETED gameweeks, so transfers
    already made for the upcoming one do not show up and the balance reads too
    high. Confirming the squad confirms the transfers that produced it."""
    p = tmp_path / "state.json"
    save_state(State(free_transfers=0, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), p)
    live, _ = resolve_current_squad(Config(entry_id=1), gw=4, state_path=p,
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.free_transfers == 0


def test_confirmed_squad_from_an_older_gameweek_is_ignored(tmp_path):
    """A squad recorded for GW4 says nothing about GW5 — by then the API's own
    picks are authoritative again."""
    p = tmp_path / "state.json"
    save_state(State(free_transfers=0, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4), p)
    live, _ = resolve_current_squad(Config(entry_id=1), gw=5, state_path=p,
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.current_squad == list(range(1, 16))  # straight from PICKS


def test_without_a_confirmed_squad_the_api_still_wins(tmp_path):
    live, _ = resolve_current_squad(Config(entry_id=1), gw=4,
                                    state_path=tmp_path / "state.json",
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.current_squad == list(range(1, 16))


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
    # Preserved, not preserved-and-incremented: the chip consumes the
    # gameweek's own free transfer, so 1 in is 1 out.
    assert s.free_transfers == 1


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


# --- Confirmation records what was applied (2026-09-07 review) ---

NEW_SQUAD = list(range(200, 215))


def test_confirming_records_the_squad_that_was_actually_applied(tmp_path):
    """Recording the transfer COUNT without the squad left state contradicting
    itself: purchase prices moved to the new players while `squad` still named
    the old ones, so a re-run before the deadline planned from the wrong 15."""
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    written = record_transfers(path, cfg, gw=4, transfers_made=1, chip=None,
                               purchase_prices={i: 5.0 for i in NEW_SQUAD},
                               squad=NEW_SQUAD, bank=0.7)
    back = load_state(path, cfg)
    assert back.squad == NEW_SQUAD
    assert back.squad_event == 4
    assert back.bank == 0.7
    assert written.squad == NEW_SQUAD


def test_confirming_without_a_squad_leaves_the_recorded_one_alone(tmp_path):
    """A Mode 1 rebuild or a legacy call says nothing about what is owned."""
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    save_state(State(free_transfers=1, last_event=3, chips_used=[],
                     squad=[1, 2, 3], squad_event=4, bank=0.2), path)
    record_transfers(path, cfg, gw=4, transfers_made=1, chip=None)
    back = load_state(path, cfg)
    assert back.squad == [1, 2, 3]
    assert back.squad_event == 4


def test_replanning_a_confirmed_gameweek_does_not_hand_back_the_accrued_transfer(tmp_path):
    """After confirming GW4, `free_transfers` is GW5's balance -- the weekly +1
    has already accrued. Re-planning GW4 with that number offered a transfer
    that was already spent, and the optimizer took it for free."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=1, free_transfers=1)
    save_state(State(free_transfers=1, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), path)
    record_transfers(path, cfg, gw=4, transfers_made=1, chip=None,
                     squad=NEW_SQUAD, bank=0.0)

    live, _ = resolve_current_squad(cfg, gw=4, state_path=path,
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.current_squad == NEW_SQUAD
    assert live.free_transfers == 0
    assert load_state(path, cfg).free_transfers == 1  # still 1 waiting for GW5


def test_replanning_a_confirmed_wildcard_gameweek_keeps_the_balance(tmp_path):
    """A Wildcard spends no free transfer, so re-planning that gameweek must
    still show the banked balance -- not zero."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=1, free_transfers=2)
    save_state(State(free_transfers=2, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), path)
    record_transfers(path, cfg, gw=4, transfers_made=11, chip="wildcard",
                     squad=NEW_SQUAD, bank=0.0)

    live, _ = resolve_current_squad(cfg, gw=4, state_path=path,
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.free_transfers == 2
    assert live.chips_used == ["wildcard"]


def test_replanning_a_confirmed_free_hit_gameweek_keeps_the_balance(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(entry_id=1, free_transfers=2)
    save_state(State(free_transfers=2, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), path)
    record_transfers(path, cfg, gw=4, transfers_made=9, chip="freehit",
                     squad=NEW_SQUAD, bank=0.0)

    live, _ = resolve_current_squad(cfg, gw=4, state_path=path,
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.free_transfers == 2
    assert live.chips_used == ["freehit"]


def test_confirming_records_the_gameweek_a_chip_was_played_in(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    record_transfers(path, cfg, gw=7, transfers_made=0, chip="benchboost")
    assert load_state(path, cfg).chip_events == [{"chip": "benchboost", "event": 7}]


def test_resolved_squad_reports_chips_fpl_says_are_already_played(tmp_path):
    """A chip played in the FPL app never touches local state. Without reading
    the API's chip list the advisor recommends it again the following week."""
    history = dict(HISTORY_NO_TRANSFERS, chips=[{"name": "3xc", "event": 1}])
    live, _ = resolve_current_squad(Config(entry_id=1), gw=2,
                                    state_path=tmp_path / "state.json",
                                    client=FakeClient(PICKS, history))
    assert live.chips_used == ["triplecaptain"]
    assert any("triplecaptain" in w for w in live.warnings)


def test_no_chips_played_means_no_chip_warning(tmp_path):
    live, _ = resolve_current_squad(Config(entry_id=1), gw=2,
                                    state_path=tmp_path / "state.json",
                                    client=FakeClient(PICKS, HISTORY_NO_TRANSFERS))
    assert live.chips_used == []
    assert live.warnings == []


def test_confirming_persists_a_chip_fpl_played_outside_this_tool(tmp_path):
    """A chip played in the FPL app reaches the advisor through the API merge;
    it has to survive into local state too, or the next confirmation drops it."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=1, free_transfers=1)
    history = dict(HISTORY_NO_TRANSFERS, chips=[{"name": "bboost", "event": 1}])
    live, _ = resolve_current_squad(cfg, gw=2, state_path=path,
                                    client=FakeClient(PICKS, history))
    record_transfers(path, cfg, gw=2, transfers_made=1, chip=None,
                     squad=NEW_SQUAD, bank=0.0, api_chips=live.chip_events)
    back = load_state(path, cfg)
    assert back.chips_used == ["benchboost"]
    assert back.chip_events == [{"chip": "benchboost", "event": 1}]


# --- P0/B3: a Free Hit must not destroy the permanent squad (2026-09-17 audit) ---

def test_a_free_hit_preserves_the_permanent_squad_and_its_prices(tmp_path):
    """FPL restores the pre-chip squad, bank and purchase prices at the next
    deadline. The local state held the ONLY copy of the purchase prices, and
    --confirm was overwriting it with the temporary Free Hit 15."""
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    permanent = list(range(1, 16))
    temporary = list(range(101, 116))
    prices = {i: 5.0 for i in permanent}

    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     purchase_prices={i: 6.0 for i in temporary},
                     squad=temporary, bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices=prices)

    written = load_state(path, cfg)
    assert written.squad == temporary          # this week you field the FH 15
    assert written.freehit_event == 8
    assert written.base_squad == permanent     # and you keep the real one
    assert written.base_bank == 1.2
    assert written.base_purchase_prices == prices


def test_the_gameweek_after_a_free_hit_plans_from_the_restored_squad(tmp_path):
    """resolve_current_squad reads the PREVIOUS gameweek's picks, which after a
    Free Hit is the temporary team FPL has already taken away."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=7, free_transfers=1)
    permanent = list(range(1, 16))
    temporary = list(range(101, 116))
    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     purchase_prices={i: 6.0 for i in temporary},
                     squad=temporary, bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in permanent})

    picks = {"entry_history": {"bank": 3},
             "picks": [{"element": i} for i in temporary]}
    client = FakeClient(picks, {"current": [], "chips": []})
    live, errors = resolve_current_squad(cfg, 9, path, client)

    assert errors == []
    assert live.current_squad == permanent
    assert live.bank == 1.2
    assert live.purchase_prices == {i: 5.0 for i in permanent}
    assert live.restored_from_freehit is True
    assert any("Free Hit" in w for w in live.warnings)


def test_an_ordinary_confirmation_clears_the_free_hit_marker(tmp_path):
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     squad=list(range(101, 116)), bank=0.3,
                     base_squad=list(range(1, 16)), base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in range(1, 16)})
    record_transfers(path, cfg, gw=9, transfers_made=1, chip=None,
                     squad=list(range(1, 16)), bank=1.2)
    written = load_state(path, cfg)
    assert written.freehit_event is None
    assert written.base_squad == []


# --- P0/B4: confirming a chip has to be deliberate (2026-09-17 audit) ---

def test_confirming_without_naming_the_chip_is_refused(monkeypatch, capsys):
    """A run is a proposal. Defaulting --confirm to the advised chip spends a
    chip on advice the manager may never have acted on -- and for a Free Hit it
    also decides whether the permanent squad is preserved."""
    import run_gameweek
    from fpl.optimize.chips import ChipAdvice
    from fpl.report.weekly import Recommendation

    class _Lineup:
        xi, bench, formation, captain, vice, xp = [1], [2], "4-4-2", 1, 2, 10.0

    squad = list(range(1, 16))
    rec = Recommendation(gw=5, deadline="x", mode=2, lineup=_Lineup(),
                         squad_ids=squad, chip=ChipAdvice("triplecaptain", "why"))

    monkeypatch.setattr(run_gameweek, "resolve_current_squad",
                        lambda *a, **k: (run_gameweek_live(squad), []))
    monkeypatch.setattr(run_gameweek, "run", lambda *a, **k: (rec, _fake_xp(squad)))
    monkeypatch.setattr(run_gameweek, "render", lambda *a, **k: "")
    monkeypatch.setattr(run_gameweek, "FplClient", lambda *a, **k: object())
    monkeypatch.setattr(run_gameweek, "load_overrides", lambda *a, **k: {})

    code = run_gameweek.main(["--mode", "2", "--gw", "5", "--confirm", "--no-refresh"])
    assert code == 1
    assert "--applied-chip" in capsys.readouterr().out


def run_gameweek_live(squad):
    from fpl.cli import LiveSquad
    return LiveSquad(squad, 0.0, 1, [], {}, [], [])


def _fake_xp(squad):
    import pandas as pd
    return pd.DataFrame({"player_id": squad, "price": [5.0] * len(squad)})


# --- RB4: a same-gameweek Free Hit re-confirmation must not corrupt the base ---

def test_reconfirming_a_free_hit_keeps_the_original_permanent_squad(tmp_path):
    """On a same-GW re-run resolve_current_squad returns the already-confirmed
    TEMPORARY squad, and the CLI hands that back as base_squad. The chip record
    was idempotent; the base was not, and a second --confirm overwrote the only
    copy of the real fifteen with the one-week team."""
    path = tmp_path / "state.json"
    cfg = Config(free_transfers=1)
    permanent = list(range(1, 16))
    temporary = list(range(101, 116))

    record_transfers(path, cfg, gw=6, transfers_made=9, chip="freehit",
                     squad=temporary, bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in permanent})
    # Second confirmation of the SAME Free Hit, with the temporary squad
    # resupplied as the base -- exactly what run_gameweek.py does on a re-run.
    record_transfers(path, cfg, gw=6, transfers_made=0, chip="freehit",
                     squad=temporary, bank=0.3,
                     base_squad=temporary, base_bank=0.3,
                     base_purchase_prices={i: 6.0 for i in temporary})

    written = load_state(path, cfg)
    assert written.base_squad == permanent
    assert written.base_bank == 1.2
    assert written.base_purchase_prices == {i: 5.0 for i in permanent}


# --- RB9: confirmation refuses unknown or illegal chips ---

def _confirm_harness(monkeypatch, squad, advised_chip=None, chip_events=()):
    """Stub everything around run_gameweek.main so only the confirm path runs."""
    import run_gameweek
    from fpl.cli import LiveSquad
    from fpl.optimize.chips import ChipAdvice
    from fpl.report.weekly import Recommendation

    class _Lineup:
        xi, bench, formation, captain, vice, xp = [1], [2], "4-4-2", 1, 2, 10.0

    rec = Recommendation(gw=8, deadline="2099-01-01T00:00:00Z", mode=2,
                         lineup=_Lineup(), squad_ids=squad,
                         chip=ChipAdvice(advised_chip, "why"))
    live = LiveSquad(squad, 0.0, 1, [], {}, [], list(chip_events))
    written = {}
    monkeypatch.setattr(run_gameweek, "resolve_current_squad", lambda *a, **k: (live, []))
    monkeypatch.setattr(run_gameweek, "run", lambda *a, **k: (rec, _fake_xp(squad)))
    monkeypatch.setattr(run_gameweek, "render", lambda *a, **k: "")
    monkeypatch.setattr(run_gameweek, "FplClient", lambda *a, **k: object())
    monkeypatch.setattr(run_gameweek, "load_overrides", lambda *a, **k: {})
    monkeypatch.setattr(run_gameweek, "mark_actioned", lambda *a, **k: None)

    def fake_record(*a, **k):
        written["called"] = True
        from fpl.state import State
        return State()
    monkeypatch.setattr(run_gameweek, "record_transfers", fake_record)
    return run_gameweek, written


def test_an_unknown_chip_name_is_rejected_by_argparse(monkeypatch, capsys):
    run_gameweek, _ = _confirm_harness(monkeypatch, list(range(1, 16)))
    with pytest.raises(SystemExit):
        run_gameweek.main(["--mode", "2", "--gw", "8", "--confirm", "--no-refresh",
                           "--applied-chip", "wildcrad"])


def test_a_second_same_window_chip_is_refused_at_confirmation(monkeypatch, capsys):
    """The advisor and the replay both refuse this; confirmation used to persist
    it, and every later run then reasoned from a chip history that never
    happened."""
    run_gameweek, written = _confirm_harness(
        monkeypatch, list(range(1, 16)),
        chip_events=[{"chip": "wildcard", "event": 4}])
    code = run_gameweek.main(["--mode", "2", "--gw", "8", "--confirm", "--no-refresh",
                              "--applied-chip", "wildcard"])
    assert code == 1
    assert "Refusing" in capsys.readouterr().out
    assert "called" not in written


def test_reconfirming_the_same_chip_in_the_same_gameweek_is_allowed(monkeypatch):
    run_gameweek, written = _confirm_harness(
        monkeypatch, list(range(1, 16)),
        chip_events=[{"chip": "wildcard", "event": 8}])
    code = run_gameweek.main(["--mode", "2", "--gw", "8", "--confirm", "--no-refresh",
                              "--applied-chip", "wildcard"])
    assert code == 0
    assert written.get("called")


# --- RB11: Free Hit restoration works without the picks endpoint ---

def test_free_hit_restoration_does_not_need_the_picks_fetch(tmp_path):
    """Everything needed to restore ownership and bank is in local state, and
    the picks endpoint would only return the temporary squad anyway."""
    path = tmp_path / "state.json"
    cfg = Config(entry_id=7, free_transfers=1)
    permanent = list(range(1, 16))
    record_transfers(path, cfg, gw=8, transfers_made=9, chip="freehit",
                     squad=list(range(101, 116)), bank=0.3,
                     base_squad=permanent, base_bank=1.2,
                     base_purchase_prices={i: 5.0 for i in permanent})
    client = FakeClient(picks_error=RuntimeError("503"),
                        history={"current": [], "chips": []})
    live, errors = resolve_current_squad(cfg, 9, path, client)
    assert errors == []
    assert live.current_squad == permanent
    assert live.bank == 1.2
    assert live.restored_from_freehit is True


def test_a_confirmed_squad_does_not_need_the_picks_fetch_either(tmp_path):
    p = tmp_path / "state.json"
    save_state(State(free_transfers=0, last_event=3, chips_used=[],
                     squad=list(range(100, 115)), squad_event=4, bank=0.0), p)
    live, errors = resolve_current_squad(
        Config(entry_id=1), gw=4, state_path=p,
        client=FakeClient(picks_error=RuntimeError("503"), history=HISTORY_NO_TRANSFERS))
    assert errors == []
    assert live.current_squad == list(range(100, 115))


# --- RR1: a pre-deadline confirmation marks the PLANNING forecast ---

def test_a_pre_deadline_confirmation_marks_the_planning_forecast(monkeypatch, tmp_path):
    """--confirm re-runs the whole pipeline and writes a new forecast first.
    Before this fix that new forecast, being the newest pre-deadline version,
    was marked as the one acted on instead of the planning forecast the
    manager had actually looked at."""
    import run_gameweek
    from datetime import datetime, timezone
    from fpl.backtest import manifest

    deadline = "2099-01-01T00:00:00Z"
    root = tmp_path / "data"
    manifest.record_version(root, gw=8, version="plan", origin="live",
                            created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
                            deadline=deadline)

    squad = list(range(1, 16))
    run_gameweek_mod, written = _confirm_harness(monkeypatch, squad)

    # The stubbed pipeline "writes" its own forecast, exactly as the real one does.
    real_run = run_gameweek_mod.run
    def run_and_record(*a, **k):
        manifest.record_version(root, gw=8, version="confirm-rerun", origin="live",
                                created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
                                deadline=deadline)
        return real_run(*a, **k)
    monkeypatch.setattr(run_gameweek_mod, "run", run_and_record)
    monkeypatch.setattr(run_gameweek_mod, "ROOT", tmp_path)
    monkeypatch.setattr(run_gameweek_mod, "mark_actioned", manifest.mark_actioned)

    code = run_gameweek_mod.main(["--mode", "2", "--gw", "8", "--confirm",
                                  "--no-refresh", "--applied-chip", "none",
                                  "--config", str(_REPO / "config.yaml")])
    assert code == 0
    assert manifest.select_version(root, 8)["version"] == "plan"


def test_forecast_version_flag_pins_the_marked_version(monkeypatch, tmp_path):
    import run_gameweek
    from datetime import datetime, timezone
    from fpl.backtest import manifest

    deadline = "2099-01-01T00:00:00Z"
    root = tmp_path / "data"
    for i, day in enumerate((9, 10)):
        manifest.record_version(root, gw=8, version=f"v{i}", origin="live",
                                created_at=datetime(2026, 9, day, tzinfo=timezone.utc),
                                deadline=deadline)
    run_gameweek_mod, _ = _confirm_harness(monkeypatch, list(range(1, 16)))
    monkeypatch.setattr(run_gameweek_mod, "ROOT", tmp_path)
    monkeypatch.setattr(run_gameweek_mod, "mark_actioned", manifest.mark_actioned)
    code = run_gameweek_mod.main(["--mode", "2", "--gw", "8", "--confirm", "--no-refresh",
                                  "--applied-chip", "none", "--forecast-version", "v0",
                                  "--config", str(_REPO / "config.yaml")])
    assert code == 0
    assert manifest.select_version(root, 8)["version"] == "v0"
