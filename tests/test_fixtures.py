import math
import pandas as pd
from fpl.model.fixtures import team_fixture_frame, fixture_counts

RATINGS = pd.DataFrame({
    "team_id": [1, 2, 3],
    "att": [1.5, 1.0, 0.5],
    "dfn": [0.5, 1.0, 1.5],
    "confidence": ["high", "high", "high"],
})

FIX = pd.DataFrame({
    "fixture_id": [101, 102, 103, 104],
    "event": [1, 2, 2, 3],
    "team_h": [1, 1, 2, 2],
    "team_a": [2, 3, 3, 1],
    "team_h_difficulty": [3, 2, 2, 4],
    "team_a_difficulty": [4, 5, 4, 2],
    "kickoff_time": ["2026-08-21T19:00:00Z"] * 4,
    "finished": [False] * 4,
})


def test_each_fixture_yields_two_team_rows():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    assert len(df) == 8
    assert set(df[df.fixture_id == 101].team_id) == {1, 2}


def test_home_flag_and_opponent_resolved():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    row = df[(df.fixture_id == 101) & (df.team_id == 1)].iloc[0]
    assert bool(row.is_home) is True
    assert row.opponent_id == 2


def test_clean_sheet_probability_is_exp_neg_xgc():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    row = df[(df.fixture_id == 101) & (df.team_id == 1)].iloc[0]
    assert math.isclose(row.p_cs, math.exp(-row.xgc), rel_tol=1e-9)
    assert 0 < row.p_cs < 1


def test_strong_defence_vs_weak_attack_has_higher_cs_probability():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3).set_index(
        ["fixture_id", "team_id"])
    # team 1 (dfn 0.5) at home vs team 2 attack; team 2 (dfn 1.0) away vs team 1 attack
    assert df.loc[(101, 1), "p_cs"] > df.loc[(101, 2), "p_cs"]


def test_horizon_filters_events():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=1)
    assert set(df.event) == {1}


def test_fixture_counts_detects_double_gameweek():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3).set_index(
        ["team_id", "event"])
    assert counts.loc[(1, 2), "n_fixtures"] == 1
    assert counts.loc[(3, 2), "n_fixtures"] == 2  # team 3 plays twice in event 2


def test_fixture_counts_detects_blank_gameweek():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3).set_index(
        ["team_id", "event"])
    assert counts.loc[(3, 1), "n_fixtures"] == 0  # team 3 has no event-1 fixture
    assert counts.loc[(3, 3), "n_fixtures"] == 0


def test_fixture_counts_covers_every_team_event_pair():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3)
    assert len(counts) == 9  # 3 teams x 3 events
