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
# Promoted: no PL history (zero minutes). Filler1/2 exist only to give the
# league mean enough established teams that "Average" centres near 1.0 —
# a 2-team sample (Strong + Average alone) skews the mean too far for the
# 0.9-1.1 tolerance below; this is a fixture-realism fix, not a formula change.
# Minutes are a realistic squad-season (~36k; a 20-man squad over 38 games
# records ~65k) because ratings are now shrunk toward the strength_overall
# prior by how much football backs them. At the old fixture value of 3,000 an
# "established" team would sit at a quarter weight and read as low confidence.
PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "team_id": [1, 2, 3, 4, 5],
    "goals_scored": [70, 50, 0, 50, 50],
    "goals_conceded": [32, 50, 0, 50, 50],
    "minutes": [36000, 36000, 0, 36000, 36000],
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


# --- team ratings on a per-90 scale, shrunk by squad minutes ---

# Two teams that defend identically per 90, one having played four times the
# minutes. Rating on raw totals made the busier squad look four times worse.
PER90_TEAMS = pd.DataFrame({
    "team_id": [1, 2, 3, 4],
    "name": ["Busy", "Quiet", "Filler1", "Filler2"],
    "short_name": ["BSY", "QUI", "F1", "F2"],
    "strength_overall_home": [3, 3, 3, 3],
    "strength_overall_away": [3, 3, 3, 3],
})
PER90_PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "team_id": [1, 2, 3, 4],
    "goals_scored": [400, 100, 400, 400],
    "goals_conceded": [400, 100, 400, 400],
    "minutes": [36000, 9000, 36000, 36000],
})


def test_ratings_use_per_90_rates_not_season_totals():
    """Hull City were rated the best defence in the league because their squad
    had 1,477 Premier League minutes on record against Arsenal's 41,084 — the
    rating measured how much football a squad had played, not how well. Two
    squads conceding at the same rate must rate the same."""
    r = team_ratings(PER90_PLAYERS, PER90_TEAMS).set_index("team_id")
    # Same goals per 90 (1.0), four times the minutes.
    assert r.loc[1, "dfn"] == pytest.approx(r.loc[2, "dfn"], abs=0.25)
    assert r.loc[1, "att"] == pytest.approx(r.loc[2, "att"], abs=0.25)


def test_thin_squad_is_shrunk_toward_its_overall_strength_prior():
    """A promoted club with a handful of recorded minutes must not be handed an
    elite rating on that evidence, however good the per-90 rate looks."""
    teams = PER90_TEAMS.copy()
    teams.loc[teams.team_id == 2, ["strength_overall_home",
                                   "strength_overall_away"]] = 1
    players = PER90_PLAYERS.copy()
    # Team 2: a superb per-90 record built from almost nothing.
    players.loc[players.team_id == 2, ["goals_conceded", "minutes"]] = [2, 900]
    r = team_ratings(players, teams).set_index("team_id")
    measured = (2 / (900 / 90)) / (400 / (36000 / 90))  # 0.2 of league rate
    assert r.loc[2, "dfn"] > measured * 2, "thin evidence was trusted outright"
    assert r.loc[2, "confidence"] != "high"


def test_established_squad_keeps_its_measured_rating():
    r = team_ratings(PER90_PLAYERS, PER90_TEAMS).set_index("team_id")
    assert r.loc[1, "confidence"] == "high"
    assert r.loc[1, "dfn"] == pytest.approx(1.0, abs=0.2)


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
