"""Monte Carlo points distributions, with the correlation xP cannot express.

`model.xp` answers "how many points, on average". That is the right question
for a squad you are trying to maximise the expected score of, and the wrong
one for a league you are trying to WIN: rank is decided by the shape of the
distribution and by how much of it you share with everyone else. Two things
follow that a per-player expectation cannot represent:

* Teammates keep the SAME clean sheet. Three defenders from one club are one
  bet placed three times, not three bets, and an expectation-only model prices
  them identically either way.
* A player with a fat tail is worth more for rank than a metronome with the
  same mean, because rank is won by hauls and not by averages.

So the match is drawn rather than integrated: one draw of the actual scoring
events per simulation, shared within a team-fixture where the real thing is
shared. Means agree with `model.xp` by construction -- every rate here is the
same rate, and `tests/test_simulate.py` pins the agreement -- so this adds a
distribution without moving the projection underneath it.
"""
import numpy as np
import pandas as pd

from .minutes import M_START, M_SUB
from .xp import (GOAL_PTS, CS_PTS, ASSIST_PTS, DC_PTS, DC_THRESHOLD,
                 SAVES_PER_POINT, CONCEDED_PER_PENALTY, CONCEDED_PENALTY_POSITIONS)
from .bps import FIXTURE_SENSITIVITY, MAX_BONUS_PER_MATCH

DEFAULT_SIMS = 4000
# Dispersion of a team's attacking output around its expectation, as the shape
# of a mean-1 Gamma multiplier shared by everyone in the side. It leaves each
# player's MEAN untouched (E[Poisson(lambda*theta)] = lambda) while making
# teammates' returns move together: the days a side scores three are the days
# several of its attackers return. Lower shape = heavier team-level swings.
TEAM_FORM_SHAPE = 4.0


def _aligned(frame: pd.DataFrame, ids, columns) -> np.ndarray:
    """Columns of `frame` as a float matrix, row-ordered to match `ids`."""
    return frame.set_index("player_id").reindex(ids)[columns].astype(float).to_numpy()


def simulate_event(players: pd.DataFrame, rates: pd.DataFrame, minutes: pd.DataFrame,
                   tfx: pd.DataFrame, event: int, n_sims: int = DEFAULT_SIMS,
                   seed: int = 0) -> tuple[list[int], np.ndarray]:
    """Draw `n_sims` gameweek outcomes for every player.

    Returns (player_ids, samples) with `samples` shaped (n_players, n_sims) in
    FPL points. A team with no fixture in `event` scores zero everywhere, and a
    team with two is summed over both -- blanks and doubles fall out of the
    fixture loop exactly as they do in `model.xp`.
    """
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in players["player_id"]]
    row_of = {pid: i for i, pid in enumerate(ids)}
    samples = np.zeros((len(ids), n_sims), dtype=float)

    rate_cols = ["xg90", "xa90", "bonus90", "dc90", "saves90", "cards90"]
    mins_cols = ["p_start", "p_play", "p_60", "m_start", "e_minutes"]
    R = _aligned(rates, ids, rate_cols)
    has_m_start = "m_start" in minutes.columns
    M = _aligned(minutes, ids, mins_cols if has_m_start
                 else [c for c in mins_cols if c != "m_start"])
    if not has_m_start:  # frames from outside model.minutes
        M = np.insert(M, 3, M[:, 2] * 0 + M_START, axis=1)
    positions = players.set_index("player_id").reindex(ids)["position"].to_numpy()
    team_of = players.set_index("player_id").reindex(ids)["team_id"].astype(int).to_numpy()

    for _, fx in tfx[tfx["event"] == int(event)].iterrows():
        rows = np.flatnonzero(team_of == int(fx["team_id"]))
        if len(rows) == 0:
            continue
        samples[rows] += _simulate_fixture(
            R[rows], M[rows], positions[rows], fx, n_sims, rng)
    return ids, samples


def _simulate_fixture(R, M, positions, fx, n_sims, rng) -> np.ndarray:
    """One team's points in one fixture, shape (n_players, n_sims)."""
    n = len(R)
    xg90, xa90, bonus90, dc90, saves90, cards90 = (R[:, i][:, None] for i in range(6))
    p_start, p_play, p_60, m_start = (M[:, i][:, None] for i in range(4))

    # --- who is on the pitch -------------------------------------------------
    # One uniform per player-sim splits start / substitute / absent, so the
    # three outcomes stay mutually exclusive and sum to p_play.
    u = rng.random((n, n_sims))
    started = u < p_start
    subbed = (u >= p_start) & (u < p_play)
    played = started | subbed
    mins = np.where(started, m_start, np.where(subbed, M_SUB, 0.0))
    # Reaching the hour is drawn separately rather than read off `mins`:
    # `m_start` is a player's AVERAGE start length, so thresholding it would
    # make every start either always or never reach 60. p_60/p_start is the
    # model's own conditional probability.
    with np.errstate(divide="ignore", invalid="ignore"):
        p60_given_start = np.where(p_start > 0, p_60 / np.maximum(p_start, 1e-12), 0.0)
    reached_60 = started & (rng.random((n, n_sims)) < np.clip(p60_given_start, 0.0, 1.0))
    share = mins / 90.0

    # --- the shared half of the match ---------------------------------------
    # Goals conceded is drawn ONCE for the side. This is the correlation the
    # whole module exists for: every defender's clean sheet is the same event.
    conceded_team = rng.poisson(float(fx["xgc"]), size=n_sims)[None, :]
    # Goals conceded while a given player was on the pitch. Thinning a Poisson
    # by time-on-pitch is again Poisson, so each player's MARGINAL matches
    # model.xp exactly while remaining tied to his teammates'.
    conceded_on = rng.binomial(np.broadcast_to(conceded_team, (n, n_sims)),
                               np.clip(share, 0.0, 1.0))
    # A mean-1 team attacking multiplier, shared like the clean sheet.
    theta = rng.gamma(TEAM_FORM_SHAPE, 1.0 / TEAM_FORM_SHAPE, size=n_sims)[None, :]

    att = float(fx["att_mult"])
    goals = rng.poisson(np.maximum(xg90 * share * att * theta, 0.0))
    assists = rng.poisson(np.maximum(xa90 * share * att * theta, 0.0))
    dc = rng.poisson(np.maximum(dc90 * share, 0.0))
    threat = float(fx["opp_threat"]) if "opp_threat" in fx else 1.0
    saves = rng.poisson(np.maximum(saves90 * share * threat, 0.0))
    cards = rng.poisson(np.maximum(cards90 * share, 0.0))
    # Bonus is 0-3 whole points. A binomial with the model's expected bonus as
    # its mean keeps `model.xp` intact while giving the tail somewhere to go.
    scale = 1.0 + (att - 1.0) * FIXTURE_SENSITIVITY
    b = np.clip(bonus90 * share * scale, 0.0, MAX_BONUS_PER_MATCH)
    bonus = rng.binomial(int(MAX_BONUS_PER_MATCH), b / MAX_BONUS_PER_MATCH)

    # --- the scoring rules ---------------------------------------------------
    goal_pts = np.array([GOAL_PTS[p] for p in positions])[:, None]
    cs_pts = np.array([CS_PTS[p] for p in positions])[:, None]
    dc_bar = np.array([DC_THRESHOLD.get(p, 12) for p in positions])[:, None]
    is_keeper = (positions == "GKP")[:, None]
    concedes = np.isin(positions, list(CONCEDED_PENALTY_POSITIONS))[:, None]

    pts = played * 1.0 + reached_60 * 1.0
    pts += goals * goal_pts + assists * ASSIST_PTS
    # A clean sheet is the SIDE keeping the ball out for the whole match, not
    # only while this player was on it, and it needs the hour.
    pts += (conceded_team == 0) * reached_60 * cs_pts
    pts += (dc >= dc_bar) * DC_PTS
    pts += bonus
    pts -= concedes * (conceded_on // CONCEDED_PER_PENALTY)
    pts += is_keeper * (saves // SAVES_PER_POINT)
    pts -= cards
    return pts.astype(float)
