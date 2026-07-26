"""Per-fixture difficulty, clean-sheet probability, and DGW/BGW detection."""
import numpy as np
import pandas as pd
from .strength import HOME_ATT, AWAY_ATT


def team_fixture_frame(fixtures: pd.DataFrame, ratings: pd.DataFrame,
                       from_event: int, horizon: int) -> pd.DataFrame:
    events = range(from_event, from_event + horizon)
    fx = fixtures[fixtures["event"].isin(events)]
    r = ratings.set_index("team_id")

    rows = []
    for _, f in fx.iterrows():
        for team_id, opp_id, is_home in (
            (f["team_h"], f["team_a"], True),
            (f["team_a"], f["team_h"], False),
        ):
            if team_id not in r.index or opp_id not in r.index:
                continue
            venue = HOME_ATT if is_home else AWAY_ATT
            xgc = float(r.loc[team_id, "dfn"]) * float(r.loc[opp_id, "att"]) / venue
            rows.append({
                "team_id": int(team_id),
                "event": int(f["event"]),
                "fixture_id": int(f["fixture_id"]),
                "opponent_id": int(opp_id),
                "is_home": is_home,
                "xgc": xgc,
                "p_cs": float(np.exp(-xgc)),
                "att_mult": float(r.loc[team_id, "att"]) * float(r.loc[opp_id, "dfn"]) * venue,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


def fixture_counts(fixtures: pd.DataFrame, team_ids, from_event: int,
                   horizon: int) -> pd.DataFrame:
    events = list(range(from_event, from_event + horizon))
    fx = fixtures[fixtures["event"].isin(events)]
    played = {}
    for _, f in fx.iterrows():
        for t in (f["team_h"], f["team_a"]):
            played[(int(t), int(f["event"]))] = played.get((int(t), int(f["event"])), 0) + 1
    rows = [
        {"team_id": int(t), "event": int(e), "n_fixtures": played.get((int(t), int(e)), 0)}
        for t in team_ids for e in events
    ]
    return pd.DataFrame(rows).reset_index(drop=True)
