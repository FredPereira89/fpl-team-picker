"""Raw FPL JSON -> tidy DataFrames. Resolves IDs, converts price, flags availability."""
import pandas as pd

AVAILABLE = "a"
FLOAT_COLS = ["expected_goals", "expected_assists", "expected_goals_conceded"]
# Who takes the set pieces, as published by FPL. Null for everyone not on the
# list, and 1 for the first-choice taker. Left nullable on purpose: filling the
# gap with 0 would sort every non-taker ahead of the designated one.
SET_PIECE_COLS = ["penalties_order", "corners_and_indirect_freekicks_order",
                  "direct_freekicks_order"]
PLAYER_INT_COLS = [
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "yellow_cards", "red_cards", "own_goals",
    "defensive_contribution",
]
PAST_COLS = PLAYER_INT_COLS + [
    "clearances_blocks_interceptions", "tackles", "recoveries",
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
    for c in SET_PIECE_COLS:
        # Absent from older cached snapshots; that reads the same as "not a taker".
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    cols = ["player_id", "web_name", "team_id", "team", "position", "price", "status",
            "available", "news", "chance_of_playing"] + PLAYER_INT_COLS + FLOAT_COLS + \
           SET_PIECE_COLS + ["selected_by_percent"]
    return df[cols].reset_index(drop=True)


# The counting stats the model reads as a full-season baseline. bootstrap-static
# reports these as CURRENT-season cumulative totals, which the FPL API resets to
# zero at each season rollover -- so from GW1 onward they must be sourced from a
# completed season instead. Everything outside this list (price, status, news,
# team, position, ownership) describes today and still comes from bootstrap.
BASELINE_COLS = PLAYER_INT_COLS + FLOAT_COLS


def latest_season(past: pd.DataFrame) -> str | None:
    """Most recently completed season in a `history_past_frame`, or None if empty.

    Season names sort correctly as strings ("2024/25" < "2025/26").
    """
    if past is None or len(past) == 0:
        return None
    names = sorted(str(s) for s in past["season_name"].unique())
    return names[-1] if names else None


def apply_season_baseline(players: pd.DataFrame, past: pd.DataFrame,
                          season: str | None = None) -> pd.DataFrame:
    """Swap season-to-date bootstrap totals for a completed season's totals.

    `fpl.model.scoring.per90_rates` and `fpl.model.minutes.minutes_model` both
    assume `players` carries a full season of history (they shrink toward a
    positional mean with k=900 minutes and divide `starts` by 38). That holds
    pre-season, when bootstrap still shows last season's totals, and breaks the
    moment GW1 kicks off: every established player collapses to a ~90-minute
    sample while players who did not feature look better via the price prior.

    A player with no row for `season` is zeroed, which routes them to the
    price prior in `minutes_model` -- the correct treatment for a newcomer.
    """
    out = players.copy()
    season = season or latest_season(past)
    src = None
    if season is not None:
        rows = past[past["season_name"].astype(str) == str(season)]
        if len(rows):
            src = rows.drop_duplicates("player_id").set_index("player_id")

    for c in BASELINE_COLS:
        if src is not None and c in src.columns:
            vals = out["player_id"].map(src[c])
        else:
            vals = pd.Series(index=out.index, dtype="float64")
        vals = pd.to_numeric(vals, errors="coerce").fillna(0)
        out[c] = vals.astype(int) if c in PLAYER_INT_COLS else vals.astype(float)

    return out[list(players.columns)].reset_index(drop=True)


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
