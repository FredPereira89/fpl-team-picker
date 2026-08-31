import pytest
import pandas as pd
from fpl.model.strength import (team_ratings, league_goals_per_team_match,
                                HOME_ATT, AWAY_ATT, DEFAULT_LEAGUE_GC)

TEAMS = pd.DataFrame({
    "team_id": [1, 2, 3, 4, 5],
    "name": ["Strong", "Average", "Promoted", "Filler1", "Filler2"],
    "short_name": ["STR", "AVG", "PRO", "FIL1", "FIL2"],
    "strength_overall_home": [5, 3, 1, 3, 3],
    "strength_overall_away": [5, 3, 1, 3, 3],
})

# Strong: 70 goals for, 32 against. Average/Filler1/Filler2: identical 50/50.
# Promoted: no PL history (zero minutes). Minutes are a whole squad's season
# (~34,000) rather than one player's, because the established/promoted test is
# on squad minutes -- see MIN_TEAM_MINUTES. Filler1/2 exist only to give the
# league mean enough established teams that "Average" centres near 1.0 —
# a 2-team sample (Strong + Average alone) skews the mean too far for the
# 0.9-1.1 tolerance below; this is a fixture-realism fix, not a formula change.
PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "team_id": [1, 2, 3, 4, 5],
    "goals_scored": [70, 50, 0, 50, 50],
    "goals_conceded": [32, 50, 0, 50, 50],
    "minutes": [34000, 34000, 0, 34000, 34000],
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


def test_promoted_team_with_a_few_experienced_signings_still_falls_back():
    """A handful of Premier League minutes is not a Premier League season.

    Coventry came up in 2026 with five players carrying 2,415 PL minutes
    between them for their old clubs: 6 goals scored, 42 conceded. Read as if
    it were a full season, that is an attack of 0.13 and a defensive leak of
    0.09 — the best defence in the league by a factor of ten. `MCI v COV` then
    came out as Man City's HARDEST attacking fixture of GW3 (att_mult 0.18)
    and the optimizer tried to sell Haaland on a -8 hit.

    The team qualifies as established on its squad's minutes, so a club whose
    sample is a few players' part-seasons takes the strength prior instead.
    """
    partial = PLAYERS.copy()
    partial.loc[partial["team_id"] == 3, ["goals_scored", "goals_conceded", "minutes"]] = [6, 42, 2415]
    r = team_ratings(partial, TEAMS).set_index("team_id")

    assert r.loc[3, "confidence"] == "low"
    # The specific inversion that caused the bug: a promoted club rated as a
    # better defence than the best established team in the league.
    assert r.loc[3, "dfn"] > r.loc[1, "dfn"]


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


# --- P2: goals-conceded scale (2026-08-27 audit) ---

def test_league_goals_per_team_match_uses_per_90_not_season_totals():
    """A first-choice keeper who missed games still concedes at the team's rate.

    Dividing a team's season total by 38 undercounts, because no single player
    is on the pitch for all 38 matches.
    """
    players = pd.DataFrame({
        "player_id": [1, 2],
        "position": ["GKP", "GKP"],
        "minutes": [2700.0, 2700.0],   # 30 matches
        "goals_conceded": [45, 36],    # 1.5 and 1.2 per 90
    })
    assert league_goals_per_team_match(players, min_minutes=900) == pytest.approx(1.35)


def test_league_goals_per_team_match_ignores_small_samples():
    players = pd.DataFrame({
        "player_id": [1, 2],
        "position": ["GKP", "DEF"],
        "minutes": [2700.0, 90.0],
        "goals_conceded": [45, 5],  # 5.0 per 90 from a single match
    })
    assert league_goals_per_team_match(players, min_minutes=900) == pytest.approx(1.5)


def test_league_goals_per_team_match_falls_back_when_no_history():
    empty = pd.DataFrame({"player_id": [], "position": [], "minutes": [],
                          "goals_conceded": []})
    assert league_goals_per_team_match(empty) == DEFAULT_LEAGUE_GC
