"""Assemble expected points per player.

THIS MODULE DEFINES THE MODEL/OPTIMIZER CONTRACT. optimize/ consumes only
CONTRACT_COLUMNS. Any replacement model that emits this frame is a drop-in.

xP is computed per fixture and summed over the fixtures in an event, so
double gameweeks (2+ fixtures) and blanks (0 fixtures) fall out for free.
"""
import pandas as pd
from scipy.stats import poisson

from .bps import expected_bonus

# Always present. A frame ALSO carries one `xp_gw{event}` column per gameweek
# in the horizon (see EVENT_PREFIX in optimize.objective): the optimizers use
# them to move the armband week by week, which a single horizon total cannot
# express, and the ledger keeps them so a scored gameweek can be traced back to
# the fixture-by-fixture projection that produced it.
CONTRACT_COLUMNS = [
    "player_id", "web_name", "team", "position", "price",
    "xp_next1", "xp_next5", "xp_horizon", "p_start", "p_play", "e_minutes",
    "confidence", "flags",
]
EVENT_PREFIX = "xp_gw"

GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
DC_PTS = 2
DC_THRESHOLD = {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12}
SAVES_PER_POINT = 3
CONCEDED_PER_PENALTY = 2
CONCEDED_PENALTY_POSITIONS = {"GKP", "DEF"}
# Where to stop summing the threshold tail. Ten points' worth of saves (30) or
# conceded goals (20) in one match has probability far below rounding.
MAX_THRESHOLDS = 10


def p_dc_threshold(dc90: float, e_minutes: float, position: str) -> float:
    """Probability of hitting the Defensive Contribution threshold in one match."""
    if e_minutes <= 0 or dc90 <= 0:
        return 0.0
    threshold = DC_THRESHOLD.get(position, 12)
    lam = float(dc90) * float(e_minutes) / 90.0
    return float(poisson.sf(threshold - 1, lam))


def expected_thresholds(lam: float, per_point: int) -> float:
    """E[floor(N / per_point)] for a Poisson count N.

    FPL awards saves and deducts conceded goals in WHOLE steps -- one point per
    three saves, minus one per two conceded -- so the expectation of the award
    is not the award on the expectation. E[N]/k overstates save points by about
    a third of a point per match and overstates the conceded deduction by about
    a quarter, in opposite directions, which is exactly the shape of the
    goalkeeper bias in the score ledger.

    Uses E[floor(N/k)] = sum_{m>=1} P(N >= k*m).
    """
    lam = float(lam)
    if lam <= 0:
        return 0.0
    return float(sum(poisson.sf(per_point * m - 1, lam)
                     for m in range(1, MAX_THRESHOLDS + 1)))


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
        pts -= expected_thresholds(float(fx_row["xgc"]) * share, CONCEDED_PER_PENALTY)
    if position == "GKP":
        # Saves are the opponent's shots on target, so they scale with how much
        # shooting this fixture invites -- not with the keeper's own history
        # alone. `opp_threat` is this fixture's expected goals conceded relative
        # to an average one.
        threat = float(fx_row["opp_threat"]) if "opp_threat" in fx_row else 1.0
        pts += expected_thresholds(float(rate_row["saves90"]) * share * threat,
                                   SAVES_PER_POINT)
    pts -= float(rate_row["cards90"]) * share
    return max(0.0, pts)


def build_xp(players: pd.DataFrame, rates: pd.DataFrame, minutes: pd.DataFrame,
             tfx: pd.DataFrame, counts: pd.DataFrame, cfg, from_event: int) -> pd.DataFrame:
    r = rates.set_index("player_id")
    m = minutes.set_index("player_id")
    horizon_events = list(range(from_event, from_event + cfg.horizon_gw))
    decay = float(getattr(cfg, "horizon_decay", 1.0))

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
            # xp_next5 is the honest total a human is shown. xp_horizon is the
            # same points discounted by cfg.horizon_decay per gameweek and is
            # what the optimizers maximise: a gain five weeks out is worth less
            # than one this week, because the squad can be changed before then
            # and the projection is far less certain.
            "xp_next5": round(sum(per_event.values()), 4),
            "xp_horizon": round(sum(
                v * decay ** (event - from_event) for event, v in per_event.items()
            ), 4),
            "p_start": float(mins_row["p_start"]),
            "p_play": float(mins_row["p_play"]),
            "e_minutes": float(mins_row["e_minutes"]),
            "confidence": mins_row["confidence"],
            "flags": list(mins_row["flags"]),
            # A blank gameweek is a real zero, not a missing value, so every
            # horizon event gets a column whether or not the team plays.
            **{f"{EVENT_PREFIX}{e}": round(per_event.get(e, 0.0), 4)
               for e in horizon_events},
        })
    columns = CONTRACT_COLUMNS + [f"{EVENT_PREFIX}{e}" for e in horizon_events]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)
