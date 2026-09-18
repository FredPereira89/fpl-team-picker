"""Assemble expected points per player.

THIS MODULE DEFINES THE MODEL/OPTIMIZER CONTRACT. optimize/ consumes only
CONTRACT_COLUMNS. Any replacement model that emits this frame is a drop-in.

xP is computed per fixture and summed over the fixtures in an event, so
double gameweeks (2+ fixtures) and blanks (0 fixtures) fall out for free.
"""
import pandas as pd
from scipy.stats import poisson

from .bps import expected_bonus_for
from .minutes import M_START, M_SUB

# Always present. A frame ALSO carries one `xp_gw{event}` column per gameweek
# in the horizon (see EVENT_PREFIX in optimize.objective): the optimizers use
# them to move the armband week by week, which a single horizon total cannot
# express, and the ledger keeps them so a scored gameweek can be traced back to
# the fixture-by-fixture projection that produced it.
CONTRACT_COLUMNS = [
    "player_id", "web_name", "team", "position", "price",
    "xp_next1", "xp_next5", "xp_horizon", "p_start", "p_play", "p_60", "e_minutes",
    # Percent of FPL managers who own him. Leagues are won on RANK, so points
    # scored by a player most of the field also owns move you nowhere. The
    # optimizer needs this to express that (optimize.objective.effective_xp).
    "ownership",
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


def p_dc_threshold(dc90: float, minutes: float, position: str) -> float:
    """P(hit the Defensive Contribution threshold) GIVEN `minutes` on the pitch.

    Conditional on a known minutes figure this is exact. Feeding it a blended
    expectation is not -- see `minutes_branches`.
    """
    if minutes <= 0 or dc90 <= 0:
        return 0.0
    threshold = DC_THRESHOLD.get(position, 12)
    lam = float(dc90) * float(minutes) / 90.0
    return float(poisson.sf(threshold - 1, lam))


def minutes_branches(mins_row) -> list[tuple[float, float]]:
    """[(probability, minutes)] over the outcomes one match can actually take.

    A player starts and plays `m_start`, or comes off the bench for `M_SUB`, or
    does not feature. Every THRESHOLD award -- Defensive Contribution, saves,
    goals conceded -- is a nonlinear function of minutes, so its expectation
    has to be taken over this distribution rather than evaluated once at
    `e_minutes`. Evaluating at the mean is Jensen's inequality applied
    backwards and understates DC by ~0.13 pts/match for a rotation-risk
    defender: it prices a player as though he reliably played 73 minutes, when
    he in fact plays 80 or none.

    The linear terms (goals, assists, cards) are correctly priced at
    `e_minutes` and deliberately still are -- for those, expectation and
    evaluation-at-the-mean coincide.
    """
    # Frames built outside model.minutes (older ledger parquets, hand-built
    # rows) carry only e_minutes. Treat those as a certainty at that many
    # minutes, which is exactly how they priced before this existed.
    if "p_start" not in mins_row:
        return [(1.0, float(mins_row["e_minutes"]))]
    p_start = float(mins_row["p_start"])
    p_play = float(mins_row["p_play"])
    # Same reasoning for m_start: the league-average start length stands in.
    m_start = float(mins_row["m_start"]) if "m_start" in mins_row else M_START
    p_sub = max(0.0, p_play - p_start)
    return [(p_start, m_start), (p_sub, M_SUB)]


def p_dc_threshold_mixture(dc90: float, mins_row, position: str) -> float:
    """P(hit the DC threshold), integrated over the minutes distribution."""
    return float(sum(p * p_dc_threshold(dc90, m, position)
                     for p, m in minutes_branches(mins_row) if p > 0))


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


def expected_thresholds_over_minutes(rate90: float, mins_row, per_point: int) -> float:
    """E[floor(N / per_point)] integrated over the minutes distribution.

    Same correction as `p_dc_threshold_mixture`, for the two stepped awards
    that scale with time on the pitch: saves earned and goals conceded.
    """
    return float(sum(p * expected_thresholds(rate90 * m / 90.0, per_point)
                     for p, m in minutes_branches(mins_row) if p > 0))


def team_goal_scales(players: pd.DataFrame, rates: pd.DataFrame,
                     minutes: pd.DataFrame, tfx: pd.DataFrame) -> dict:
    """{(team_id, fixture_id): factor} capping player goals at the team total.

    Each player's historical xG per 90 already reflects the attack he plays
    in, and the fixture multiplier then scales it by his club's attack rating
    again -- so an elite attack's players were counted twice and, summed,
    could exceed the goals the strength model expects the side to score. The
    same rule the simulation's allocation applies (`simulate._allocate`): when
    the players' expected goals exceed the side's expected total -- the
    OPPONENT's expected goals conceded -- every attacker is scaled to fit it.
    When they fall short nothing changes; the remainder is goals by nobody in
    the frame. The strength model's team number is the authority, and the
    deterministic projection now agrees with the simulation exactly.
    """
    if "fixture_id" not in tfx.columns:
        return {}
    r = rates.set_index("player_id")
    m = minutes.set_index("player_id")
    lam = {}
    for _, p in players.iterrows():
        pid, team = int(p["player_id"]), int(p["team_id"])
        if pid not in r.index or pid not in m.index:
            continue
        w = max(float(r.loc[pid, "xg90"]), 0.0) * max(float(m.loc[pid, "e_minutes"]), 0.0) / 90.0
        lam[team] = lam.get(team, 0.0) + w
    xgc_of = {(int(f["team_id"]), int(f["fixture_id"])): float(f["xgc"])
              for _, f in tfx.iterrows()}
    out = {}
    for _, f in tfx.iterrows():
        team, fixture = int(f["team_id"]), int(f["fixture_id"])
        opp = int(f["opponent_id"]) if "opponent_id" in f and not pd.isna(f["opponent_id"]) else None
        side_total = xgc_of.get((opp, fixture)) if opp is not None else None
        players_total = lam.get(team, 0.0) * float(f["att_mult"])
        if side_total is None or players_total <= 0:
            out[(team, fixture)] = 1.0
        else:
            out[(team, fixture)] = min(1.0, side_total / players_total)
    return out


def xp_for_fixture(rate_row, mins_row, fx_row, position: str, bonus: float) -> float:
    e_min = float(mins_row["e_minutes"])
    if e_min <= 0:
        return 0.0
    share = e_min / 90.0
    p_play, p_60 = float(mins_row["p_play"]), float(mins_row["p_60"])
    # The team-total cap from `team_goal_scales`, when the caller attached it.
    goal_scale = float(fx_row["goal_scale"]) if "goal_scale" in fx_row else 1.0

    pts = p_play + p_60  # 1 pt for appearing, 2 for 60+
    pts += (float(rate_row["xg90"]) * share * float(fx_row["att_mult"]) * goal_scale
            * GOAL_PTS[position])
    pts += float(rate_row["xa90"]) * share * float(fx_row["att_mult"]) * ASSIST_PTS
    pts += float(fx_row["p_cs"]) * CS_PTS[position] * p_60
    pts += p_dc_threshold_mixture(float(rate_row["dc90"]), mins_row, position) * DC_PTS
    pts += bonus
    if position in CONCEDED_PENALTY_POSITIONS:
        pts -= expected_thresholds_over_minutes(float(fx_row["xgc"]), mins_row,
                                                CONCEDED_PER_PENALTY)
    if position == "GKP":
        # Saves are the opponent's shots on target, so they scale with how much
        # shooting this fixture invites -- not with the keeper's own history
        # alone. `opp_threat` is this fixture's expected goals conceded relative
        # to an average one.
        threat = float(fx_row["opp_threat"]) if "opp_threat" in fx_row else 1.0
        pts += expected_thresholds_over_minutes(
            float(rate_row["saves90"]) * threat, mins_row, SAVES_PER_POINT)
    pts -= float(rate_row["cards90"]) * share
    return max(0.0, pts)


def build_xp(players: pd.DataFrame, rates: pd.DataFrame, minutes: pd.DataFrame,
             tfx: pd.DataFrame, cfg, from_event: int) -> pd.DataFrame:
    r = rates.set_index("player_id")
    m = minutes.set_index("player_id")
    horizon_events = list(range(from_event, from_event + cfg.horizon_gw))
    decay = float(getattr(cfg, "horizon_decay", 1.0))
    scales = team_goal_scales(players, rates, minutes, tfx)
    if scales:
        tfx = tfx.copy()
        tfx["goal_scale"] = [scales.get((int(t), int(f)), 1.0)
                             for t, f in zip(tfx["team_id"], tfx["fixture_id"])]

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
            bonus = expected_bonus_for(rate_row["bonus90"], mins_row["e_minutes"],
                                       att_mult=float(fx["att_mult"]))
            per_event[event] = per_event.get(event, 0.0) + xp_for_fixture(
                rate_row, mins_row, fx, pos, bonus
            )

        rows.append({
            "player_id": pid,
            "web_name": p["web_name"],
            "team": p["team"],
            "position": pos,
            "price": float(p["price"]),
            "ownership": float(p.get("selected_by_percent", 0.0) or 0.0),
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
            # Kept so the ledger can score the hour as a probability forecast,
            # not only the points it feeds into.
            "p_60": float(mins_row["p_60"]) if "p_60" in mins_row else float("nan"),
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
