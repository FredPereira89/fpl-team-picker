import pandas as pd
from fpl.model.bps import expected_bonus

RATES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "bonus90": [1.2, 0.1, 0.0],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "e_minutes": [85.0, 85.0, 0.0],
})


def test_high_bps_player_earns_more_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[1] > b.loc[2]


def test_zero_minutes_earns_no_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[3] == 0.0


def test_bonus_scales_with_expected_minutes():
    half = MINUTES.copy()
    half.loc[half.player_id == 1, "e_minutes"] = 42.5
    assert expected_bonus(RATES, half).loc[1] < expected_bonus(RATES, MINUTES).loc[1]


def test_favourable_fixture_multiplier_increases_bonus():
    base = expected_bonus(RATES, MINUTES).loc[1]
    boosted = expected_bonus(RATES, MINUTES, att_mult=1.4).loc[1]
    assert boosted > base


def test_bonus_never_exceeds_three_per_match():
    hot = pd.DataFrame({"player_id": [1], "bonus90": [99.0]})
    mins = pd.DataFrame({"player_id": [1], "e_minutes": [90.0]})
    assert expected_bonus(hot, mins, att_mult=3.0).loc[1] <= 3.0


def test_returns_series_indexed_by_player_id():
    b = expected_bonus(RATES, MINUTES)
    assert b.index.name == "player_id"
    assert set(b.index) == {1, 2, 3}
