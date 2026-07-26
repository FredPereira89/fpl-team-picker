import pandas as pd
from fpl.model.strength import team_ratings, HOME_ATT, AWAY_ATT

TEAMS = pd.DataFrame({
    "team_id": [1, 2, 3, 4, 5],
    "name": ["Strong", "Average", "Promoted", "Filler1", "Filler2"],
    "short_name": ["STR", "AVG", "PRO", "FIL1", "FIL2"],
    "strength_overall_home": [5, 3, 1, 3, 3],
    "strength_overall_away": [5, 3, 1, 3, 3],
})

# Strong: 70 goals for, 32 against. Average/Filler1/Filler2: identical 50/50.
# Promoted: no PL history (zero minutes). Filler1/2 exist only to give the
# league mean enough established teams that "Average" centres near 1.0 —
# a 2-team sample (Strong + Average alone) skews the mean too far for the
# 0.9-1.1 tolerance below; this is a fixture-realism fix, not a formula change.
PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "team_id": [1, 2, 3, 4, 5],
    "goals_scored": [70, 50, 0, 50, 50],
    "goals_conceded": [32, 50, 0, 50, 50],
    "minutes": [3000, 3000, 0, 3000, 3000],
})


def test_ratings_centred_on_one():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert 0.9 < r.loc[2, "att"] < 1.1
    assert 0.9 < r.loc[2, "dfn"] < 1.1


def test_strong_team_has_higher_attack_and_lower_defence_factor():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[1, "att"] > r.loc[2, "att"]
    assert r.loc[1, "dfn"] < r.loc[2, "dfn"]


def test_promoted_team_falls_back_and_is_low_confidence():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[3, "confidence"] == "low"
    assert r.loc[3, "att"] > 0
    assert r.loc[3, "dfn"] > 0
    # weakest overall strength -> worst attack of the three
    assert r.loc[3, "att"] < r.loc[2, "att"]


def test_established_teams_are_high_confidence():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[1, "confidence"] == "high"


def test_odds_provider_overrides_promoted_rating():
    class FakeOdds:
        def team_factors(self, team_ids):
            return {3: (1.25, 0.80)}

    r = team_ratings(PLAYERS, TEAMS, odds_provider=FakeOdds()).set_index("team_id")
    assert r.loc[3, "att"] == 1.25
    assert r.loc[3, "dfn"] == 0.80
    assert r.loc[3, "confidence"] == "high"


def test_home_away_constants_are_symmetric_about_one():
    assert HOME_ATT > 1.0 > AWAY_ATT
    assert round((HOME_ATT + AWAY_ATT) / 2, 6) == 1.0
