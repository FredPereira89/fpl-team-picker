import pytest
import pandas as pd
from fpl.data.normalize import (
    normalize_teams, normalize_players, normalize_fixtures, history_past_frame,
)
from fpl.data.store import save_table, load_table

BOOTSTRAP = {
    "teams": [
        {"id": 1, "name": "Arsenal", "short_name": "ARS",
         "strength_overall_home": 4, "strength_overall_away": 5},
        {"id": 7, "name": "Liverpool", "short_name": "LIV",
         "strength_overall_home": 5, "strength_overall_away": 4},
    ],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"},
        {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"},
        {"id": 4, "singular_name_short": "FWD"},
    ],
    "elements": [
        {"id": 12, "web_name": "Saka", "team": 1, "element_type": 3, "now_cost": 95,
         "status": "a", "news": "", "chance_of_playing_next_round": None,
         "minutes": 2218, "starts": 25, "total_points": 157, "goals_scored": 7,
         "assists": 10, "clean_sheets": 12, "goals_conceded": 16, "saves": 0,
         "bonus": 18, "bps": 570, "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
         "expected_goals": "7.57", "expected_assists": "7.16",
         "expected_goals_conceded": "15.57", "selected_by_percent": "11.2",
         "defensive_contribution": 184, "penalties_order": 1,
         "corners_and_indirect_freekicks_order": 1, "direct_freekicks_order": 2},
        {"id": 99, "web_name": "Crock", "team": 7, "element_type": 2, "now_cost": 45,
         "status": "i", "news": "Knee injury - expected back 05 Sep",
         "chance_of_playing_next_round": 0,
         "minutes": 900, "starts": 10, "total_points": 40, "goals_scored": 0,
         "assists": 1, "clean_sheets": 4, "goals_conceded": 12, "saves": 0,
         "bonus": 2, "bps": 180, "yellow_cards": 1, "red_cards": 0, "own_goals": 0,
         "expected_goals": "0.40", "expected_assists": "0.90",
         "expected_goals_conceded": "13.10", "selected_by_percent": "0.4",
         "defensive_contribution": 25, "penalties_order": None,
         "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": None},
    ],
}


def test_price_converted_from_tenths():
    df = normalize_players(BOOTSTRAP)
    assert df.loc[df.player_id == 12, "price"].iloc[0] == 9.5
    assert df.loc[df.player_id == 99, "price"].iloc[0] == 4.5


def test_ids_resolved_to_names():
    df = normalize_players(BOOTSTRAP)
    row = df[df.player_id == 12].iloc[0]
    assert row["team"] == "Arsenal"
    assert row["position"] == "MID"


def test_availability_flagged_from_status():
    df = normalize_players(BOOTSTRAP).set_index("player_id")
    assert bool(df.loc[12, "available"]) is True
    assert bool(df.loc[99, "available"]) is False
    assert "Knee injury" in df.loc[99, "news"]


def test_expected_stats_are_numeric():
    df = normalize_players(BOOTSTRAP)
    assert df["expected_goals"].dtype.kind == "f"
    assert df.loc[df.player_id == 12, "expected_goals"].iloc[0] == 7.57


def test_normalize_teams_shape():
    df = normalize_teams(BOOTSTRAP)
    assert list(df.columns[:3]) == ["team_id", "name", "short_name"]
    assert len(df) == 2


def test_normalize_fixtures_keeps_difficulty():
    fx = [{"id": 1, "event": 1, "team_h": 1, "team_a": 7, "team_h_difficulty": 2,
           "team_a_difficulty": 5, "kickoff_time": "2026-08-21T19:00:00Z", "finished": False}]
    df = normalize_fixtures(fx)
    assert df.loc[0, "team_h_difficulty"] == 2
    assert df.loc[0, "event"] == 1


def test_history_past_frame_flattens_seasons():
    summaries = {12: {"history_past": [
        {"season_name": "2024/25", "total_points": 127, "minutes": 2000, "starts": 22,
         "goals_scored": 6, "assists": 9, "clean_sheets": 10, "goals_conceded": 20,
         "saves": 0, "bonus": 12, "bps": 400, "yellow_cards": 1, "red_cards": 0,
         "own_goals": 0, "expected_goals": "6.0", "expected_assists": "6.5",
         "expected_goals_conceded": "18.0", "defensive_contribution": 150,
         "clearances_blocks_interceptions": 25, "tackles": 35, "recoveries": 90},
        {"season_name": "2025/26", "total_points": 157, "minutes": 2218, "starts": 25,
         "goals_scored": 7, "assists": 10, "clean_sheets": 12, "goals_conceded": 16,
         "saves": 0, "bonus": 18, "bps": 570, "yellow_cards": 2, "red_cards": 0,
         "own_goals": 0, "expected_goals": "7.57", "expected_assists": "7.16",
         "expected_goals_conceded": "15.57", "defensive_contribution": 184,
         "clearances_blocks_interceptions": 28, "tackles": 40, "recoveries": 116},
    ]}}
    df = history_past_frame(summaries)
    assert len(df) == 2
    assert set(df.player_id) == {12}
    assert df[df.season_name == "2025/26"]["defensive_contribution"].iloc[0] == 184


def test_store_roundtrip(tmp_path):
    df = normalize_players(BOOTSTRAP)
    save_table(df, "players", tmp_path)
    out = load_table("players", tmp_path)
    assert len(out) == len(df)
    assert out.loc[out.player_id == 12, "price"].iloc[0] == 9.5


def test_defensive_contribution_extracted_for_current_season():
    """Regression: normalize_players must carry defensive_contribution through
    (needed by model/scoring.py's per90_rates dc90 rate) -- the bootstrap
    elements payload includes it as a real current-season stat.
    """
    df = normalize_players(BOOTSTRAP)
    assert df.loc[df.player_id == 12, "defensive_contribution"].iloc[0] == 184
    assert df.loc[df.player_id == 99, "defensive_contribution"].iloc[0] == 25


# --- P4: set-piece roles (2026-08-27 audit) ---

def test_normalized_players_carry_the_published_set_piece_roles():
    """FPL publishes who takes the penalties. The model read none of these."""
    df = normalize_players(BOOTSTRAP).set_index("player_id")
    assert "penalties_order" in df.columns
    assert "corners_and_indirect_freekicks_order" in df.columns
    assert "direct_freekicks_order" in df.columns


def test_a_player_with_no_set_piece_duty_is_not_treated_as_first_choice():
    """FPL sends null for everyone who is not on the list; a 0 or a NaN that
    silently sorts as 'first taker' would hand a premium to the whole league."""
    df = normalize_players(BOOTSTRAP).set_index("player_id")
    assert df["penalties_order"].isna().any()
    assert (df["penalties_order"].dropna() >= 1).all()
