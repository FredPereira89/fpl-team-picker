"""Raw FPL JSON -> tidy DataFrames. Resolves IDs, converts price, flags availability."""
import pandas as pd

AVAILABLE = "a"
FLOAT_COLS = ["expected_goals", "expected_assists", "expected_goals_conceded"]
PLAYER_INT_COLS = [
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "yellow_cards", "red_cards", "own_goals",
]
PAST_COLS = PLAYER_INT_COLS + [
    "defensive_contribution", "clearances_blocks_interceptions", "tackles", "recoveries",
]


def normalize_teams(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"])
    df = df.rename(columns={"id": "team_id"})
    return df[["team_id", "name", "short_name", "strength_overall_home", "strength_overall_away"]]


def normalize_players(bootstrap: dict) -> pd.DataFrame:
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    df = pd.DataFrame(bootstrap["elements"])
    df = df.rename(columns={"id": "player_id", "team": "team_id",
                            "chance_of_playing_next_round": "chance_of_playing"})
    df["team"] = df["team_id"].map(teams)
    df["position"] = df["element_type"].map(positions)
    df["price"] = df["now_cost"] / 10.0
    df["available"] = df["status"] == AVAILABLE
    df["news"] = df["news"].fillna("")
    for c in FLOAT_COLS + ["selected_by_percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in PLAYER_INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    cols = ["player_id", "web_name", "team_id", "team", "position", "price", "status",
            "available", "news", "chance_of_playing"] + PLAYER_INT_COLS + FLOAT_COLS + \
           ["selected_by_percent"]
    return df[cols].reset_index(drop=True)


def normalize_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(fixtures).rename(columns={"id": "fixture_id"})
    cols = ["fixture_id", "event", "team_h", "team_a", "team_h_difficulty",
            "team_a_difficulty", "kickoff_time", "finished"]
    return df[cols].reset_index(drop=True)


def history_past_frame(summaries: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for pid, summary in summaries.items():
        for season in summary.get("history_past", []):
            row = {"player_id": pid, "season_name": season["season_name"]}
            for c in PAST_COLS:
                row[c] = pd.to_numeric(season.get(c, 0), errors="coerce")
            for c in FLOAT_COLS:
                row[c] = pd.to_numeric(season.get(c, 0.0), errors="coerce")
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["player_id", "season_name"] + PAST_COLS + FLOAT_COLS)
    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
