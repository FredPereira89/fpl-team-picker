import numpy as np
import pandas as pd
import pytest
from fpl.config import Config
from fpl.model.xp import (build_xp, p_dc_threshold, p_dc_threshold_mixture,
                          xp_for_fixture, minutes_branches, expected_thresholds,
                          expected_thresholds_over_minutes, CONTRACT_COLUMNS,
                          feasible_goal_and_assist_marginals)

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
    return build_xp(PLAYERS, RATES, MINUTES, TFX, CFG, from_event=1)


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
    single = build_xp(PLAYERS, RATES, mins, TFX, CFG, from_event=2)
    alpha = single.set_index("player_id").loc[2, "xp_next1"]  # Alpha keeper, 1 fixture
    beta = single.set_index("player_id").loc[3, "xp_next1"]   # Beta mid, 2 fixtures
    per_fixture = build_xp(PLAYERS, RATES, mins, TFX, CFG, from_event=3)
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
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, cfg, from_event=1).set_index("player_id")
    # Alpha plays once per event with identical fixtures, so each event's xP is equal
    per_event = df.loc[1, "xp_next1"]
    expected = per_event * sum(0.5 ** n for n in range(5))
    assert df.loc[1, "xp_horizon"] == pytest.approx(expected, abs=1e-3)


def test_undiscounted_total_is_still_reported_for_the_user():
    """xp_next5 is what the report shows a human. It must stay a real
    points total, not a discounted score that only the solver understands."""
    cfg = Config(horizon_gw=5, horizon_decay=0.5)
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, cfg, from_event=1).set_index("player_id")
    assert df.loc[1, "xp_next5"] == pytest.approx(df.loc[1, "xp_next1"] * 5, abs=1e-3)
    assert df.loc[1, "xp_horizon"] < df.loc[1, "xp_next5"]


def test_no_decay_leaves_the_horizon_score_equal_to_the_total():
    cfg = Config(horizon_gw=5, horizon_decay=1.0)
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, cfg, from_event=1).set_index("player_id")
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


# --- threshold points must integrate over the minutes distribution ---------
# A match is 90 minutes or 0. Feeding blended `e_minutes` into a Poisson tail
# is Jensen's inequality applied backwards -- the same error `expected_thresholds`
# exists to fix for saves, left in place one function above it.

def _mins(p_start, m_start, p_play=None):
    return pd.Series({
        "p_start": p_start, "p_play": p_start if p_play is None else p_play,
        "p_60": p_start * 0.9, "m_start": m_start,
        "e_minutes": p_start * m_start, "confidence": "high", "flags": [],
    })


def test_minutes_branches_cover_every_outcome_of_the_match():
    from fpl.model.minutes import M_SUB
    branches = minutes_branches(_mins(0.9, 80.0, p_play=0.935))
    assert sum(p for p, _ in branches) == pytest.approx(0.935)
    assert sum(p * m for p, m in branches) == pytest.approx(0.9 * 80.0 + 0.035 * M_SUB)


def test_dc_probability_is_the_start_sub_mixture_not_the_mean_minutes_estimate():
    """0.90 * P(N>=10 | 80 mins) + 0.035 * P(N>=10 | 20 mins), never
    P(N>=10 | 72.7 mins), which understates it by about 0.13 pts a match."""
    from scipy.stats import poisson
    from fpl.model.minutes import M_SUB
    m = _mins(0.9, 80.0, p_play=0.935)
    expected = (0.9 * poisson.sf(9, 10.5 * 80.0 / 90)
                + 0.035 * poisson.sf(9, 10.5 * M_SUB / 90))
    assert p_dc_threshold_mixture(10.5, m, "DEF") == pytest.approx(expected)
    assert p_dc_threshold_mixture(10.5, m, "DEF") > p_dc_threshold(10.5, 72.7, "DEF")


def test_two_players_with_equal_expected_minutes_differ_on_thresholds():
    """Both average 60 minutes. The one who starts 80 three weeks in four clears
    a 10-action bar far more often than the one who plays 60 every week."""
    steady = _mins(1.0, 60.0)
    rotated = _mins(0.75, 80.0)
    assert steady["e_minutes"] == pytest.approx(rotated["e_minutes"])
    assert (p_dc_threshold_mixture(10.5, rotated, "DEF")
            > p_dc_threshold_mixture(10.5, steady, "DEF"))


def test_save_points_use_the_minutes_mixture():
    from fpl.model.minutes import M_SUB
    m = _mins(0.9, 80.0, p_play=0.935)
    expected = (0.9 * expected_thresholds(3.2 * 80.0 / 90, 3)
                + 0.035 * expected_thresholds(3.2 * M_SUB / 90, 3))
    assert expected_thresholds_over_minutes(3.2, m, 3) == pytest.approx(expected)


def test_a_player_who_never_starts_scores_no_threshold_points():
    assert p_dc_threshold_mixture(10.5, _mins(0.0, 80.0), "DEF") == 0.0
    assert expected_thresholds_over_minutes(3.2, _mins(0.0, 80.0), 3) == 0.0


def test_frames_without_m_start_fall_back_to_the_league_start_length():
    """Anything built outside model.minutes (tests, older ledger frames) must
    still price, using the default start length rather than crashing."""
    from fpl.model.minutes import M_START
    legacy = pd.Series({"p_start": 0.9, "p_play": 0.935, "p_60": 0.81,
                        "e_minutes": 72.7, "confidence": "high", "flags": []})
    assert minutes_branches(legacy)[0] == pytest.approx((0.9, M_START))


def test_a_frame_without_p_start_is_treated_as_certain_minutes():
    """Frames from outside model.minutes (older ledger parquets, hand-built test
    rows) carry only e_minutes. Those must price as a certainty at that many
    minutes -- the behaviour before the mixture existed -- not crash."""
    legacy = {"e_minutes": 90.0, "p_play": 1.0, "p_60": 1.0}
    assert minutes_branches(legacy) == [(1.0, 90.0)]
    assert p_dc_threshold_mixture(10.5, legacy, "DEF") == pytest.approx(
        p_dc_threshold(10.5, 90.0, "DEF"))


# --- ownership reaches the optimizer (2026-09-09 audit) --------------------
# FPL leagues are won on RANK, not on points. Points scored by a player 60% of
# the field also owns move you nowhere; the objective was pure expected points
# and could not express that. `selected_by_percent` was normalized out of
# bootstrap and then dropped before the optimizer ever saw it.

def test_xp_frame_carries_ownership():
    players = PLAYERS.assign(selected_by_percent=[55.0, 3.0, 0.4])
    df = build_xp(players, RATES, MINUTES, TFX, CFG, from_event=1).set_index("player_id")
    assert "ownership" in df.columns
    assert df.loc[1, "ownership"] == pytest.approx(55.0)


def test_a_frame_without_ownership_reads_as_zero_not_missing():
    """Older bootstraps and hand-built frames must still price."""
    df = build_xp(PLAYERS, RATES, MINUTES, TFX, CFG, from_event=1).set_index("player_id")
    assert df.loc[1, "ownership"] == 0.0


# --- R7: player goals are capped at the team total the strength model gives ---

def test_an_over_full_attack_is_scaled_to_the_teams_expected_goals():
    """Individual xG per 90 already reflects the attack a player is in, and the
    fixture multiplier scaled it by the club's attack rating again. Summed over
    a strong side that exceeded the goals the strength model expected it to
    score. The team number wins, exactly as the simulation's allocation does."""
    import pandas as pd
    from fpl.model.xp import team_goal_scales
    players = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [1, 1, 2],
                            "position": ["FWD", "MID", "DEF"]})
    rates = pd.DataFrame({"player_id": [1, 2, 3], "xg90": [0.9, 0.6, 0.05]})
    minutes = pd.DataFrame({"player_id": [1, 2, 3], "e_minutes": [90.0, 90.0, 90.0]})
    tfx = pd.DataFrame([
        {"team_id": 1, "fixture_id": 7, "opponent_id": 2, "xgc": 0.8, "att_mult": 1.0},
        {"team_id": 2, "fixture_id": 7, "opponent_id": 1, "xgc": 1.2, "att_mult": 1.0},
    ])
    scales = team_goal_scales(players, rates, minutes, tfx)
    # Team 1's players sum to 1.5 xG; the strength model expects 1.2 (team 2's xgc).
    assert scales[(1, 7)] == pytest.approx(1.2 / 1.5)
    # Team 2's lone defender is far below the 0.8 the side is expected to score.
    assert scales[(2, 7)] == 1.0


def test_the_cap_reaches_the_projection():
    import pandas as pd
    from fpl.model.xp import xp_for_fixture
    rate = pd.Series({"xg90": 1.0, "xa90": 0.0, "bonus90": 0.0, "dc90": 0.0,
                      "saves90": 0.0, "cards90": 0.0})
    mins = pd.Series({"e_minutes": 90.0, "p_play": 1.0, "p_60": 1.0,
                      "p_start": 1.0, "m_start": 90.0})
    base = {"att_mult": 1.0, "p_cs": 0.0, "xgc": 0.0}
    full = xp_for_fixture(rate, mins, pd.Series(base), "FWD", 0.0)
    halved = xp_for_fixture(rate, mins, pd.Series({**base, "goal_scale": 0.5}), "FWD", 0.0)
    assert full - halved == pytest.approx(0.5 * 1.0 * 4)      # half a goal at 4 pts


def test_self_assist_feasibility_reaches_deterministic_xp():
    """xG/xA collisions must be constrained in both model paths.

    With A at 0.8 xG/xA and B at 0.2, a one-goal side cannot give A an
    0.8 assist probability: A scores 80% of its goals.  The feasible assist
    marginals are 0.2 for each player, which are worth 0.6 xP apiece.
    """
    goal, assist = feasible_goal_and_assist_marginals(
        np.array([0.8, 0.2]), np.array([0.8, 0.2]), team_goal_rate=1.0)
    assert goal == pytest.approx([0.8, 0.2])
    assert assist == pytest.approx([0.2, 0.2])

    players = pd.DataFrame({
        "player_id": [1, 2], "web_name": ["A", "B"],
        "team": ["Alpha", "Alpha"], "team_id": [1, 1],
        "position": ["FWD", "FWD"], "price": [7.0, 7.0],
    })
    rates = pd.DataFrame({"player_id": [1, 2], "xg90": [0.8, 0.2],
                          "xa90": [0.8, 0.2], "bonus90": [0.0, 0.0],
                          "dc90": [0.0, 0.0], "saves90": [0.0, 0.0],
                          "cards90": [0.0, 0.0]})
    minutes = pd.DataFrame({"player_id": [1, 2], "p_start": [0.0, 0.0],
                            "p_play": [0.0, 0.0], "p_60": [0.0, 0.0],
                            "m_start": [90.0, 90.0], "e_minutes": [90.0, 90.0],
                            "confidence": ["high", "high"], "flags": [[], []]})
    tfx = pd.DataFrame({"team_id": [1], "event": [1], "fixture_id": [1],
                        "is_home": [True], "xgc": [1.0], "p_cs": [0.0],
                        "att_mult": [1.0]})
    xp = build_xp(players, rates, minutes, tfx, Config(horizon_gw=1), 1).set_index("player_id")
    assert xp.loc[1, "xp_next1"] == pytest.approx(0.8 * 4 + 0.2 * 3)
    assert xp.loc[2, "xp_next1"] == pytest.approx(0.2 * 4 + 0.2 * 3)
