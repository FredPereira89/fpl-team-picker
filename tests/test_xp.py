import pandas as pd
import pytest
from fpl.config import Config
from fpl.model.xp import build_xp, p_dc_threshold, CONTRACT_COLUMNS

CFG = Config(horizon_gw=5)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3],
    "web_name": ["Striker", "Keeper", "Benched"],
    "team": ["Alpha", "Alpha", "Beta"],
    "team_id": [1, 1, 2],
    "position": ["FWD", "GKP", "MID"],
    "price": [11.0, 5.0, 4.5],
})
RATES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "xg90": [0.7, 0.0, 0.05],
    "xa90": [0.3, 0.0, 0.05],
    "bonus90": [0.8, 0.3, 0.0],
    "dc90": [1.0, 0.0, 3.0],
    "saves90": [0.0, 3.0, 0.0],
    "cards90": [0.1, 0.05, 0.1],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "p_start": [0.95, 0.9, 0.0],
    "p_play": [0.97, 0.92, 0.0],
    "p_60": [0.95, 0.9, 0.0],
    "e_minutes": [82.0, 76.0, 0.0],
    "confidence": ["high", "high", "low"],
    "flags": [[], [], ["Unavailable (i): knee"]],
})
# Alpha plays every event; Beta blanks in event 1 and doubles in event 2.
TFX = pd.DataFrame({
    "team_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "event":   [1, 2, 3, 4, 5, 2, 2, 3, 4, 5],
    "fixture_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "opponent_id": [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    "is_home": [True, False, True, False, True, True, False, True, False, True],
    "xgc": [0.8] * 10,
    "p_cs": [0.45] * 10,
    "att_mult": [1.1] * 10,
})
COUNTS = pd.DataFrame(
    [{"team_id": 1, "event": e, "n_fixtures": 1} for e in range(1, 6)] +
    [{"team_id": 2, "event": 1, "n_fixtures": 0}, {"team_id": 2, "event": 2, "n_fixtures": 2}] +
    [{"team_id": 2, "event": e, "n_fixtures": 1} for e in range(3, 6)]
)


def _build():
    return build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, CFG, from_event=1)


def test_contract_columns_exact():
    """The fixed contract comes first, then one column per horizon gameweek."""
    assert list(_build().columns) == CONTRACT_COLUMNS + [f"xp_gw{e}" for e in range(1, 6)]


def test_per_gameweek_columns_sum_to_the_horizon_total():
    """The armband moves week to week, so the optimizer needs the split -- and
    it has to reconcile with the total the report shows."""
    df = _build().set_index("player_id")
    for pid in df.index:
        parts = sum(float(df.loc[pid, f"xp_gw{e}"]) for e in range(1, 6))
        assert parts == pytest.approx(float(df.loc[pid, "xp_next5"]), abs=1e-3)


def test_a_blank_gameweek_is_a_zero_not_a_gap():
    """Beta has no fixture in event 1. A missing column would make the frame
    ragged; a NaN would poison the objective."""
    df = _build().set_index("player_id")
    assert float(df.loc[3, "xp_gw1"]) == 0.0


def test_every_player_present():
    assert set(_build().player_id) == {1, 2, 3}


def test_striker_outscores_bench_player():
    df = _build().set_index("player_id")
    assert df.loc[1, "xp_next1"] > df.loc[3, "xp_next1"]


def test_zero_minutes_player_scores_zero():
    df = _build().set_index("player_id")
    assert df.loc[3, "xp_next1"] == 0.0
    assert df.loc[3, "xp_next5"] == 0.0


def test_blank_gameweek_gives_zero_for_that_event():
    # player 3 is on Beta, which has no event-1 fixture
    df = _build().set_index("player_id")
    assert df.loc[3, "xp_next1"] == 0.0


def test_double_gameweek_sums_both_fixtures():
    # Give the Beta player real minutes so the double is visible
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 3, ["p_start", "p_play", "p_60", "e_minutes"]] = [0.9, 0.95, 0.9, 80.0]
    single = build_xp(PLAYERS, RATES, mins, TFX, COUNTS, CFG, from_event=2)
    alpha = single.set_index("player_id").loc[2, "xp_next1"]  # Alpha keeper, 1 fixture
    beta = single.set_index("player_id").loc[3, "xp_next1"]   # Beta mid, 2 fixtures
    per_fixture = build_xp(PLAYERS, RATES, mins, TFX, COUNTS, CFG, from_event=3)
    beta_single = per_fixture.set_index("player_id").loc[3, "xp_next1"]
    assert beta == pytest.approx(2 * beta_single, rel=0.02)
    assert alpha > 0


def test_xp_next5_at_least_xp_next1():
    df = _build()
    assert (df["xp_next5"] >= df["xp_next1"] - 1e-9).all()


def test_flags_propagate_from_minutes_model():
    df = _build().set_index("player_id")
    assert any("Unavailable" in f for f in df.loc[3, "flags"])


def test_keeper_earns_save_points():
    df = _build().set_index("player_id")
    assert df.loc[2, "xp_next1"] > 2.0  # appearance + saves + clean-sheet share


def test_dc_threshold_uses_position_specific_bar():
    # identical rate: defenders need 10, midfielders need 12 -> defender more likely
    assert p_dc_threshold(12.0, 90.0, "DEF") > p_dc_threshold(12.0, 90.0, "MID")


def test_dc_threshold_zero_when_no_minutes():
    assert p_dc_threshold(12.0, 0.0, "DEF") == 0.0


def test_dc_threshold_is_a_probability():
    for rate in (0.0, 5.0, 20.0):
        p = p_dc_threshold(rate, 90.0, "MID")
        assert 0.0 <= p <= 1.0


# --- P5a: horizon decay (2026-08-27 audit) ---

def test_horizon_score_discounts_later_gameweeks():
    """A gain five weeks out is not worth the same as one this week: the squad
    can be changed before then, and the projection is far less certain."""
    cfg = Config(horizon_gw=5, horizon_decay=0.5)
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, cfg, from_event=1).set_index("player_id")
    # Alpha plays once per event with identical fixtures, so each event's xP is equal
    per_event = df.loc[1, "xp_next1"]
    expected = per_event * sum(0.5 ** n for n in range(5))
    assert df.loc[1, "xp_horizon"] == pytest.approx(expected, abs=1e-3)


def test_undiscounted_total_is_still_reported_for_the_user():
    """xp_next5 is what the report shows a human. It must stay a real
    points total, not a discounted score that only the solver understands."""
    cfg = Config(horizon_gw=5, horizon_decay=0.5)
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, cfg, from_event=1).set_index("player_id")
    assert df.loc[1, "xp_next5"] == pytest.approx(df.loc[1, "xp_next1"] * 5, abs=1e-3)
    assert df.loc[1, "xp_horizon"] < df.loc[1, "xp_next5"]


def test_no_decay_leaves_the_horizon_score_equal_to_the_total():
    cfg = Config(horizon_gw=5, horizon_decay=1.0)
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, cfg, from_event=1).set_index("player_id")
    assert df.loc[1, "xp_horizon"] == pytest.approx(df.loc[1, "xp_next5"], rel=1e-9)


# --- Threshold scoring for keepers and defenders (2026-09-07 review) ---

def test_save_points_are_awarded_in_whole_threes_not_continuously():
    """FPL pays 1 point per THREE saves, so a keeper on an expected 2.7 saves
    is not owed 0.9 points -- he is owed the probability of reaching 3, plus 6,
    and so on. The continuous form over-credited every keeper by about a third
    of a point a match."""
    from fpl.model.xp import expected_thresholds
    lam = 2.7
    assert expected_thresholds(lam, 3) < lam / 3
    assert expected_thresholds(lam, 3) == pytest.approx(0.565, abs=0.01)


def test_conceded_deductions_are_also_stepped():
    """-1 per TWO conceded. The continuous form over-punished defences."""
    from fpl.model.xp import expected_thresholds
    lam = 1.2
    assert expected_thresholds(lam, 2) < lam / 2


def test_no_saves_expected_means_no_save_points():
    from fpl.model.xp import expected_thresholds
    assert expected_thresholds(0.0, 3) == 0.0


def test_a_keeper_facing_a_bigger_threat_is_worth_more_saves():
    """Saves are the opponent's shots. Carrying a save rate forward unadjusted
    rated a keeper facing the best attack in the league exactly like one facing
    the worst."""
    from fpl.model.xp import xp_for_fixture
    rate = {"xg90": 0.0, "xa90": 0.0, "dc90": 0.0, "saves90": 3.0, "cards90": 0.0}
    mins = {"e_minutes": 90.0, "p_play": 1.0, "p_60": 1.0}
    quiet = {"att_mult": 1.0, "p_cs": 0.3, "xgc": 1.2, "opp_threat": 0.6}
    busy = dict(quiet, opp_threat=1.6)
    assert xp_for_fixture(rate, mins, busy, "GKP", bonus=0.0) > \
        xp_for_fixture(rate, mins, quiet, "GKP", bonus=0.0)
