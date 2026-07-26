"""Assemble expected points per player.

THIS MODULE DEFINES THE MODEL/OPTIMIZER CONTRACT. optimize/ consumes only
CONTRACT_COLUMNS. Any replacement model that emits this frame is a drop-in.

xP is computed per fixture and summed over the fixtures in an event, so
double gameweeks (2+ fixtures) and blanks (0 fixtures) fall out for free.
"""
import pandas as pd
from scipy.stats import poisson

from .bps import expected_bonus

CONTRACT_COLUMNS = [
    "player_id", "web_name", "team", "position", "price",
    "xp_next1", "xp_next5", "p_start", "e_minutes", "confidence", "flags",
]

GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
DC_PTS = 2
DC_THRESHOLD = {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12}
SAVES_PER_POINT = 3.0
CONCEDED_PENALTY_POSITIONS = {"GKP", "DEF"}


def p_dc_threshold(dc90: float, e_minutes: float, position: str) -> float:
    """Probability of hitting the Defensive Contribution threshold in one match."""
    if e_minutes <= 0 or dc90 <= 0:
        return 0.0
    threshold = DC_THRESHOLD.get(position, 12)
    lam = float(dc90) * float(e_minutes) / 90.0
    return float(poisson.sf(threshold - 1, lam))


def xp_for_fixture(rate_row, mins_row, fx_row, position: str, bonus: float) -> float:
    e_min = float(mins_row["e_minutes"])
    if e_min <= 0:
        return 0.0
    share = e_min / 90.0
    p_play, p_60 = float(mins_row["p_play"]), float(mins_row["p_60"])

    pts = p_play + p_60  # 1 pt for appearing, 2 for 60+
    pts += float(rate_row["xg90"]) * share * float(fx_row["att_mult"]) * GOAL_PTS[position]
    pts += float(rate_row["xa90"]) * share * float(fx_row["att_mult"]) * ASSIST_PTS
    pts += float(fx_row["p_cs"]) * CS_PTS[position] * p_60
    pts += p_dc_threshold(float(rate_row["dc90"]), e_min, position) * DC_PTS
    pts += bonus
    if position in CONCEDED_PENALTY_POSITIONS:
        pts -= 0.5 * float(fx_row["xgc"]) * share
    if position == "GKP":
        pts += float(rate_row["saves90"]) * share / SAVES_PER_POINT
    pts -= float(rate_row["cards90"]) * share
    return max(0.0, pts)


def build_xp(players: pd.DataFrame, rates: pd.DataFrame, minutes: pd.DataFrame,
             tfx: pd.DataFrame, counts: pd.DataFrame, cfg, from_event: int) -> pd.DataFrame:
    r = rates.set_index("player_id")
    m = minutes.set_index("player_id")
    horizon_events = list(range(from_event, from_event + cfg.horizon_gw))

    rows = []
    for _, p in players.iterrows():
        pid, pos, team_id = int(p["player_id"]), p["position"], int(p["team_id"])
        rate_row, mins_row = r.loc[pid], m.loc[pid]
        fixtures = tfx[tfx["team_id"] == team_id]

        per_event: dict[int, float] = {}
        for _, fx in fixtures.iterrows():
            event = int(fx["event"])
            if event not in horizon_events:
                continue
            bonus = float(expected_bonus(
                rates[rates.player_id == pid],
                minutes[minutes.player_id == pid],
                att_mult=float(fx["att_mult"]),
            ).loc[pid])
            per_event[event] = per_event.get(event, 0.0) + xp_for_fixture(
                rate_row, mins_row, fx, pos, bonus
            )

        rows.append({
            "player_id": pid,
            "web_name": p["web_name"],
            "team": p["team"],
            "position": pos,
            "price": float(p["price"]),
            "xp_next1": round(per_event.get(from_event, 0.0), 4),
            "xp_next5": round(sum(per_event.values()), 4),
            "p_start": float(mins_row["p_start"]),
            "e_minutes": float(mins_row["e_minutes"]),
            "confidence": mins_row["confidence"],
            "flags": list(mins_row["flags"]),
        })
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS).reset_index(drop=True)
