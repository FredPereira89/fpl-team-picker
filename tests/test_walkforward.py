"""Walk-forward backtesting: score the model on history instead of waiting.

The ledger held ONE scored gameweek, and a single gameweek cannot distinguish
a good model from a lucky one -- the weekly edge has an SD near 15 points, so
n=3 gave a 95% CI of [-36, +39]. Everything else in the statistical review was
blocked on that. This replays past gameweeks using only the data that existed
before each one, which is the only way to get a sample without waiting years.
"""
import numpy as np
import pandas as pd
import pytest

from fpl.backtest.walkforward import (realised_score, autosub, XI_MIN,
                                      squad_ledger, weekly_edge)

# 15 players: a legal squad. Player 1 is the keeper, 20 the reserve keeper.
FRAME = pd.DataFrame([
    {"player_id": 1,  "position": "GKP", "xp_next1": 4.0, "actual": 3.0,  "minutes": 90.0},
    {"player_id": 2,  "position": "DEF", "xp_next1": 5.0, "actual": 2.0,  "minutes": 90.0},
    {"player_id": 3,  "position": "DEF", "xp_next1": 4.5, "actual": 6.0,  "minutes": 90.0},
    {"player_id": 4,  "position": "DEF", "xp_next1": 4.0, "actual": 0.0,  "minutes": 0.0},
    {"player_id": 5,  "position": "MID", "xp_next1": 6.0, "actual": 9.0,  "minutes": 90.0},
    {"player_id": 6,  "position": "MID", "xp_next1": 5.5, "actual": 2.0,  "minutes": 90.0},
    {"player_id": 7,  "position": "MID", "xp_next1": 5.0, "actual": 1.0,  "minutes": 45},
    {"player_id": 8,  "position": "MID", "xp_next1": 4.5, "actual": 5.0,  "minutes": 90.0},
    {"player_id": 9,  "position": "FWD", "xp_next1": 8.0, "actual": 12.0, "minutes": 90.0},
    {"player_id": 10, "position": "FWD", "xp_next1": 6.5, "actual": 2.0,  "minutes": 90.0},
    {"player_id": 11, "position": "FWD", "xp_next1": 3.0, "actual": 0.0,  "minutes": 0.0},
    # bench
    {"player_id": 20, "position": "GKP", "xp_next1": 2.0, "actual": 5.0,  "minutes": 90.0},
    {"player_id": 21, "position": "DEF", "xp_next1": 2.5, "actual": 7.0,  "minutes": 90.0},
    {"player_id": 22, "position": "MID", "xp_next1": 2.2, "actual": 4.0,  "minutes": 90.0},
    {"player_id": 23, "position": "FWD", "xp_next1": 1.0, "actual": 0.0,  "minutes": 0.0},
]).set_index("player_id")

SQUAD = list(FRAME.index)
XI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_a_starter_who_did_not_play_is_replaced_from_the_bench():
    """FPL substitutes automatically. A backtest that ignores autosubs charges
    the model for blanks it would never actually have taken."""
    xi, subs = autosub(SQUAD, XI, FRAME)
    assert 4 not in xi and 11 not in xi        # both blanked
    assert 21 in xi                            # first eligible outfield sub
    assert len(xi) == 11
    assert len(subs) == 2


def test_autosubs_never_break_the_formation():
    xi, _ = autosub(SQUAD, XI, FRAME)
    counts = FRAME.loc[xi, "position"].value_counts()
    assert counts.get("GKP", 0) == 1
    for pos, lo in XI_MIN.items():
        assert counts.get(pos, 0) >= lo


def test_a_bench_player_who_also_blanked_is_skipped():
    xi, _ = autosub(SQUAD, XI, FRAME)
    assert 23 not in xi                        # bench forward played 0 minutes


def test_the_armband_can_only_go_to_someone_who_started():
    """C and VC are named before the deadline, so a player substituted IN
    cannot take the armband no matter how many points he scored."""
    frame = FRAME.copy()
    frame.loc[[9, 10], ["actual", "minutes"]] = 0.0
    frame.loc[21, "actual"] = 20.0             # the sub hauls
    out = realised_score(SQUAD, XI, frame)
    assert out["captain"] in XI
    assert out["captain"] != 21


def test_the_armband_falls_to_the_vice_when_the_captain_blanks():
    frame = FRAME.copy()
    frame.loc[9, ["actual", "minutes"]] = 0.0     # top xP player blanks
    out = realised_score(SQUAD, XI, frame)
    assert out["captain"] != 9
    assert out["captain"] == 10                # the vice, named before kick-off


def test_realised_score_counts_the_captain_twice():
    out = realised_score(SQUAD, XI, FRAME)
    cap = out["captain"]
    assert cap == 9
    assert out["points"] == sum(FRAME.loc[p, "actual"] for p in out["xi"]) + 12.0


def test_the_squad_ledger_reports_the_edge_against_the_real_field():
    rows = squad_ledger({1: (60.0, 50), 2: (68.0, 81), 3: (53.0, 36)})
    assert list(rows["edge"]) == [10.0, -13.0, 17.0]


def test_the_weekly_edge_is_reported_with_an_interval_not_a_bare_mean():
    """The single most misleading thing a 3-gameweek backtest can do is print
    a mean edge with no interval. The SD of the weekly edge is near 15."""
    out = weekly_edge([0.0, -13.0, 17.0])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(4.0 / 3)
    assert out["ci_low"] < out["mean"] < out["ci_high"]
    assert out["ci_high"] - out["ci_low"] > 40      # n=3 resolves nothing
    assert out["detectable_edge"] > 15              # and says so


def test_the_edge_verdict_refuses_to_call_a_tiny_sample():
    out = weekly_edge([0.0, -13.0, 17.0])
    assert "cannot" in out["verdict"].lower() or "not" in out["verdict"].lower()


def test_a_season_of_gameweeks_narrows_the_interval():
    rng = np.random.default_rng(0)
    small = weekly_edge(list(rng.normal(5, 15, 3)))
    big = weekly_edge(list(rng.normal(5, 15, 38)))
    assert big["detectable_edge"] < small["detectable_edge"]
    assert (big["ci_high"] - big["ci_low"]) < (small["ci_high"] - small["ci_low"])


# --- the replay itself: no data from the future may reach a forecast -------

def _summaries(rounds_by_player, past_season=True):
    """element-summary shaped dicts with a stable prior season plus rounds."""
    out = {}
    for pid, rounds in rounds_by_player.items():
        hist = [{"round": r, "minutes": m, "total_points": p, "starts": 1 if m >= 60 else 0,
                 "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 1,
                 "saves": 0, "bonus": 0, "bps": 10, "yellow_cards": 0, "red_cards": 0,
                 "own_goals": 0, "expected_goals": "0.1", "expected_assists": "0.1",
                 "defensive_contribution": 3} for r, m, p in rounds]
        past = [{"season_name": "2025/26", "minutes": 2500, "total_points": 120,
                 "goals_scored": 5, "assists": 4, "clean_sheets": 8, "goals_conceded": 30,
                 "saves": 0, "bonus": 8, "bps": 400, "yellow_cards": 2, "red_cards": 0,
                 "own_goals": 0, "expected_goals": "5.0", "expected_assists": "4.0",
                 "defensive_contribution": 250, "starts": 28}] if past_season else []
        out[int(pid)] = {"history": hist, "history_past": past}
    return out


def test_a_forecast_cannot_see_the_gameweek_it_is_forecasting():
    """The property the whole harness rests on. If a replayed GW2 forecast
    changes when GW2 and GW3 results are added to the input, the backtest is
    scoring the model on its own answers."""
    from fpl.backtest.walkforward import forecast_inputs
    early = _summaries({1: [(1, 90, 6)]})
    late = _summaries({1: [(1, 90, 6), (2, 90, 25), (3, 90, 30)]})
    a = forecast_inputs(early, before_event=2)
    b = forecast_inputs(late, before_event=2)
    pd.testing.assert_frame_equal(a["current"], b["current"])
    pd.testing.assert_frame_equal(a["rounds"], b["rounds"])


def test_later_gameweeks_do_reach_a_later_forecast():
    """The complement -- proving the previous test is not passing vacuously."""
    from fpl.backtest.walkforward import forecast_inputs
    late = _summaries({1: [(1, 90, 6), (2, 90, 25), (3, 90, 30)]})
    gw2 = forecast_inputs(late, before_event=2)
    gw4 = forecast_inputs(late, before_event=4)
    assert len(gw4["rounds"]) > len(gw2["rounds"])
    assert gw4["current"]["total_points"].sum() > gw2["current"]["total_points"].sum()


def test_a_replay_never_overwrites_a_live_forecast(tmp_path):
    """A replayed forecast is contaminated -- it is built from TODAY's price,
    status and news rather than the deadline's. Letting it overwrite the real
    pre-deadline record destroys the only honest thing in the ledger."""
    from fpl.backtest.ledger import save_predictions, load_predictions
    from fpl.backtest.walkforward import replayable_gameweeks
    pred = pd.DataFrame({"player_id": [1], "web_name": ["A"], "team": ["T"],
                         "position": ["MID"], "price": [7.0], "xp_next1": [5.0],
                         "xp_next5": [25.0], "p_start": [0.9], "e_minutes": [80.0],
                         "confidence": ["high"], "flags": [[]]})
    save_predictions(pred, gw=2, root=tmp_path)
    assert replayable_gameweeks([1, 2, 3], tmp_path) == [1, 3]
    assert replayable_gameweeks([1, 2, 3], tmp_path, overwrite=True) == [1, 2, 3]
    assert load_predictions(2, tmp_path)["xp_next1"].iloc[0] == 5.0


# --- RB1: a replay must READ the snapshot, not merely notice it exists ---

def _bootstrap(cost, status, team):
    return {
        "teams": [{"id": t, "name": f"Team{t}", "short_name": f"T{t}",
                   "strength_overall_home": 3, "strength_overall_away": 3}
                  for t in (1, 2)],
        "element_types": [{"id": 1, "singular_name_short": "GKP"},
                          {"id": 2, "singular_name_short": "DEF"},
                          {"id": 3, "singular_name_short": "MID"},
                          {"id": 4, "singular_name_short": "FWD"}],
        "elements": [{
            "id": 1, "web_name": "P1", "team": team, "element_type": 3,
            "now_cost": cost, "status": status, "news": "",
            "chance_of_playing_next_round": None, "minutes": 900, "starts": 10,
            "total_points": 50, "goals_scored": 3, "assists": 2, "clean_sheets": 1,
            "goals_conceded": 10, "saves": 0, "bonus": 2, "bps": 100,
            "yellow_cards": 1, "red_cards": 0, "own_goals": 0,
            "expected_goals": "2.0", "expected_assists": "1.0",
            "expected_goals_conceded": "10.0", "selected_by_percent": "5.0",
            "defensive_contribution": 10,
        }],
        "events": [{"id": 5, "deadline_time": "2026-09-11T17:30:00Z"}],
    }


def _fixtures(event):
    return [{"id": 1, "event": event, "team_h": 1, "team_a": 2,
             "team_h_difficulty": 2, "team_a_difficulty": 3,
             "kickoff_time": "2026-09-12T14:00:00Z", "finished": True}]


def test_a_point_in_time_snapshot_is_what_the_replay_reads(tmp_path):
    """Current cache and the GW5 snapshot deliberately disagree on price,
    status, club and the fixture list; the snapshot values must win."""
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest.walkforward import gameweek_inputs

    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(55, "a", 1),
                      fixtures=_fixtures(5), deadline="2026-09-11T17:30:00Z",
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={})
    assert got["point_in_time"] is True
    assert got["source"] == "snapshot"
    row = got["players"].set_index("player_id").loc[1]
    assert row["price"] == 5.5 and row["status"] == "a" and row["team_id"] == 1
    assert list(got["fixtures"]["event"]) == [5]


def test_without_a_snapshot_the_current_cache_is_used_and_flagged(tmp_path):
    from fpl.backtest.walkforward import gameweek_inputs
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={})
    assert got["point_in_time"] is False
    assert got["source"] == "current cache"
    assert got["players"].set_index("player_id").loc[1, "price"] == 9.9


def test_a_post_deadline_snapshot_is_not_used_as_point_in_time(tmp_path):
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest.walkforward import gameweek_inputs

    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(55, "a", 1),
                      fixtures=_fixtures(5), deadline="2026-09-11T17:30:00Z",
                      captured_at=datetime(2026, 9, 20, tzinfo=timezone.utc))
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={})
    assert got["point_in_time"] is False
    assert got["players"].set_index("player_id").loc[1, "price"] == 9.9


# --- RR2: the replay reads the snapshot the ACTIONED forecast read ---

def test_the_replay_pins_the_actioned_forecasts_snapshot(tmp_path):
    """Two pre-deadline captures a day apart: the actioned forecast read the
    first, and the replay must read that one, not the newer."""
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest import manifest
    from fpl.backtest.walkforward import actioned_snapshot, gameweek_inputs

    deadline = "2026-09-11T17:30:00Z"
    first = snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(55, "a", 1),
                              fixtures=_fixtures(5), deadline=deadline,
                              captured_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(57, "a", 1),
                      fixtures=_fixtures(5), deadline=deadline,
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
    manifest.record_version(tmp_path, gw=5, version="plan", origin="live",
                            created_at=datetime(2026, 9, 9, 1, tzinfo=timezone.utc),
                            deadline=deadline, snapshot=first)

    pinned = actioned_snapshot(tmp_path, 5)
    assert pinned == first
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={}, snapshot_version=pinned)
    assert got["players"].set_index("player_id").loc[1, "price"] == 5.5
    # And without the pin the newer one would have been read -- the bug.
    loose = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                            summaries={})
    assert loose["players"].set_index("player_id").loc[1, "price"] == 5.7


# --- RR5: replay calibration is production calibration ---

def test_replay_calibration_uses_the_configured_decay(tmp_path, monkeypatch):
    import pandas as pd
    from fpl.config import Config
    from fpl.model.calibration import Calibration
    from fpl.backtest import walkforward

    cal = Calibration(intercept={"MID": 0.0}, slope={"MID": 1.0},
                      pooled_intercept=0.0, pooled_slope=1.0, n_gameweeks=5,
                      n_observations=500, n_by_position={"MID": 500}, r2=0.1)
    monkeypatch.setattr("fpl.model.calibration.fit_calibration", lambda *_: cal)
    monkeypatch.setattr("fpl.model.calibration.scored_history", lambda *_: pd.DataFrame())
    xp = pd.DataFrame({"player_id": [1], "position": ["MID"],
                       "xp_next1": [4.0], "xp_next5": [8.0], "xp_horizon": [7.0],
                       "xp_gw5": [4.0], "xp_gw6": [4.0]})
    out, note = walkforward.replay_calibration(
        xp, tmp_path, {}, 5, Config(horizon_decay=0.5, calibrate=True))
    assert note.startswith("fitted")
    assert out.loc[0, "xp_horizon"] == 4.0 + 0.5 * 4.0     # not the undiscounted 8.0


def test_replay_calibration_can_be_switched_off_independently(tmp_path):
    import pandas as pd
    from fpl.config import Config
    from fpl.backtest import walkforward
    xp = pd.DataFrame({"player_id": [1], "position": ["MID"], "xp_next1": [4.0]})
    out, note = walkforward.replay_calibration(xp, tmp_path, {}, 5, Config(calibrate=False))
    assert note == "off" and out is xp


def test_the_replay_gets_the_overrides_the_live_run_applied(tmp_path):
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest.walkforward import gameweek_inputs
    news = {1: {"p_start_override": 0.9, "note": "starts", "source": "presser"}}
    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(55, "a", 1),
                      fixtures=_fixtures(5), deadline="2026-09-11T17:30:00Z",
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc), news=news)
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6), summaries={})
    assert got["news"] == news


# --- third review, finding 1: a legacy forecast must not silently read a newer capture ---

def test_a_forecast_that_predates_snapshots_forces_a_flagged_fallback(tmp_path):
    """None used to mean both "no forecast" and "forecast without a snapshot",
    and the caller treated both as "pick the newest capture" -- so a legacy
    decision replayed against data it never saw, with no warning."""
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest import manifest
    from fpl.backtest.walkforward import actioned_snapshot, gameweek_inputs, NO_SNAPSHOT

    deadline = "2026-09-11T17:30:00Z"
    manifest.record_version(tmp_path, gw=5, version="legacy", origin="live",
                            created_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
                            deadline=deadline)                      # no snapshot
    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(57, "a", 1),
                      fixtures=_fixtures(5), deadline=deadline,
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc))

    pin = actioned_snapshot(tmp_path, 5)
    assert pin == NO_SNAPSHOT
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={}, snapshot_version=pin)
    assert got["point_in_time"] is False
    assert "current cache" in got["source"]
    assert got["players"].set_index("player_id").loc[1, "price"] == 9.9
    assert snapshots.contamination_note(tmp_path, [5], versions_by_gw={5: pin}) is not None


def test_no_forecast_at_all_still_allows_automatic_snapshot_selection(tmp_path):
    from datetime import datetime, timezone
    from fpl.data import snapshots
    from fpl.backtest.walkforward import actioned_snapshot, gameweek_inputs
    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(57, "a", 1),
                      fixtures=_fixtures(5), deadline="2026-09-11T17:30:00Z",
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert actioned_snapshot(tmp_path, 5) is None
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6),
                          summaries={}, snapshot_version=None)
    assert got["source"] == "snapshot"


# --- third review, finding 2: the archived configuration is applied ---

def test_the_replay_returns_and_applies_the_archived_config(tmp_path):
    from datetime import datetime, timezone
    from fpl.config import Config
    from fpl.data import snapshots
    from fpl.backtest.walkforward import gameweek_inputs, config_for_replay
    snapshots.capture(tmp_path, 5, bootstrap=_bootstrap(55, "a", 1),
                      fixtures=_fixtures(5), deadline="2026-09-11T17:30:00Z",
                      captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
                      config={"horizon_gw": 3, "horizon_decay": 0.7,
                              "not_a_real_field": 1})
    got = gameweek_inputs(tmp_path, 5, _bootstrap(99, "i", 2), _fixtures(6), summaries={})
    assert got["config"]["horizon_gw"] == 3
    week_cfg, changed = config_for_replay(Config(horizon_gw=5, horizon_decay=0.85),
                                          got["config"])
    assert week_cfg.horizon_gw == 3 and week_cfg.horizon_decay == 0.7
    assert changed == ["horizon_decay", "horizon_gw"]
    assert not hasattr(week_cfg, "not_a_real_field")


def test_no_archived_config_returns_an_equal_copy_not_the_shared_object():
    """The script mutates the result per week; mutating the shared fallback
    changed the baseline every later week started from."""
    from fpl.config import Config
    from fpl.backtest.walkforward import config_for_replay
    cfg = Config(horizon_gw=5, rank_sims=4000)
    out, changed = config_for_replay(cfg, {})
    assert out == cfg and out is not cfg and changed == []
    out.rank_sims = 0
    assert cfg.rank_sims != 0
