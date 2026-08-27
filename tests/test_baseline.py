"""The model's baseline must come from a completed season, not season-to-date.

`bootstrap-static` per-player counting stats are CURRENT-season cumulative and
are reset to zero at the season rollover. `fpl.model.scoring` / `fpl.model.minutes`
treat them as a full-season baseline -- true only before GW1. From GW1 onward the
baseline has to come from `element-summary` `history_past`.
"""
import pandas as pd
from fpl.data.normalize import (
    normalize_players, history_past_frame, apply_season_baseline, latest_season,
)

# Saka as bootstrap reports him AFTER GW1 of a new season: one 90-minute game.
BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS",
               "strength_overall_home": 4, "strength_overall_away": 5}],
    "element_types": [{"id": 3, "singular_name_short": "MID"},
                      {"id": 4, "singular_name_short": "FWD"}],
    "elements": [
        {"id": 12, "web_name": "Saka", "team": 1, "element_type": 3, "now_cost": 95,
         "status": "d", "news": "Knock - 75% chance", "chance_of_playing_next_round": 75,
         "minutes": 90, "starts": 1, "total_points": 6, "goals_scored": 0,
         "assists": 1, "clean_sheets": 1, "goals_conceded": 0, "saves": 0,
         "bonus": 0, "bps": 21, "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
         "expected_goals": "0.31", "expected_assists": "0.44",
         "expected_goals_conceded": "0.50", "selected_by_percent": "11.2",
         "defensive_contribution": 4},
        # A brand-new signing: no Premier League history at all.
        {"id": 77, "web_name": "Newboy", "team": 1, "element_type": 4, "now_cost": 74,
         "status": "a", "news": "", "chance_of_playing_next_round": None,
         "minutes": 0, "starts": 0, "total_points": 0, "goals_scored": 0,
         "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "saves": 0,
         "bonus": 0, "bps": 0, "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
         "expected_goals": "0.0", "expected_assists": "0.0",
         "expected_goals_conceded": "0.0", "selected_by_percent": "4.4",
         "defensive_contribution": 0},
    ],
}

SUMMARIES = {12: {"history_past": [
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
]}, 77: {"history_past": []}}


def _baselined():
    past = history_past_frame(SUMMARIES)
    return apply_season_baseline(normalize_players(BOOTSTRAP), past)


def test_latest_season_picks_most_recent_completed():
    assert latest_season(history_past_frame(SUMMARIES)) == "2025/26"


def test_latest_season_of_empty_history_is_none():
    assert latest_season(history_past_frame({})) is None


def test_baseline_replaces_season_to_date_totals():
    saka = _baselined().set_index("player_id").loc[12]
    assert saka["minutes"] == 2218, "must use last season's minutes, not GW1's 90"
    assert saka["starts"] == 25
    assert saka["defensive_contribution"] == 184
    assert saka["expected_goals"] == 7.57


def test_baseline_keeps_live_columns_from_bootstrap():
    """Price, availability and news describe TODAY -- history must not overwrite them."""
    saka = _baselined().set_index("player_id").loc[12]
    assert saka["price"] == 9.5
    assert saka["status"] == "d"
    assert saka["chance_of_playing"] == 75
    assert saka["news"] == "Knock - 75% chance"
    assert saka["team"] == "Arsenal"
    assert saka["position"] == "MID"


def test_player_with_no_history_gets_zeroed_baseline():
    """No PL history => zero minutes => minutes_model falls back to the price prior."""
    new = _baselined().set_index("player_id").loc[77]
    assert new["minutes"] == 0
    assert new["starts"] == 0


def test_baseline_preserves_row_count_and_schema():
    players = normalize_players(BOOTSTRAP)
    out = apply_season_baseline(players, history_past_frame(SUMMARIES))
    assert len(out) == len(players)
    assert list(out.columns) == list(players.columns)
