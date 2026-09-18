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
    ids, samples, _ = _simulate(players, rates, minutes, tfx, event, n_sims, seed)
    return ids, samples


def simulate_event_detailed(players, rates, minutes, tfx, event: int,
                            n_sims: int = DEFAULT_SIMS, seed: int = 0) -> dict:
    """The same draw with its parts exposed, for tests and diagnostics.

    {"ids", "samples", "played", "goals", "conceded_team", "team_of"} --
    `played` records whether each player appeared in at least one fixture in the
    gameweek.  Keeping appearance separate from points matters because a player
    can appear and score zero: captaincy passes to the vice only on NO
    appearance, and autosubs use the same rule.  `goals` is each player's goals
    per scenario and `conceded_team` the goals his SIDE conceded in that
    scenario, so the coherence property (no goal against a clean sheet in the
    same match) can be asserted directly rather than inferred.
    """
    ids, samples, detail = _simulate(players, rates, minutes, tfx, event, n_sims, seed)
    return {"ids": ids, "samples": samples, **detail}


def _simulate(players, rates, minutes, tfx, event, n_sims, seed):
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in players["player_id"]]
    samples = np.zeros((len(ids), n_sims), dtype=float)
    played_all = np.zeros((len(ids), n_sims), dtype=bool)
    goals_all = np.zeros((len(ids), n_sims), dtype=float)
    conceded_all = np.zeros((len(ids), n_sims), dtype=float)

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

    rows_in = tfx[tfx["event"] == int(event)]
    # One MATCH at a time, both sides together. Simulating each team-row on
    # its own drew a side's goals independently of what the other side
    # conceded, so an attacker could score in a scenario where the opposing
    # defenders kept a clean sheet -- an impossible outcome, and exactly the
    # covariance a rank objective built on stacks and opposing players needs.
    key = "fixture_id" if "fixture_id" in rows_in.columns else None
    groups = rows_in.groupby(key) if key else [(None, rows_in.iloc[[i]]) for i in range(len(rows_in))]
    for _, match in groups:
        sides = [fx for _, fx in match.iterrows()]
        # Who is on the pitch, drawn once per side, then the scoreline.
        pitch = {int(fx["team_id"]): _on_pitch(R, M, np.flatnonzero(team_of == int(fx["team_id"])),
                                               fx, n_sims, rng) for fx in sides}
        scored, lam_of = {}, {}
        for fx in sides:
            team = int(fx["team_id"])
            opp = next((o for o in sides if int(o["team_id"]) != team), None)
            # A side's goals are Poisson at the OPPONENT's expected goals
            # conceded -- the strength model's own number, and the one
            # model.xp's clean-sheet probability is built on. A lone row with
            # no opponent in the frame falls back to its players' own total.
            lam_of[team] = (float(opp["xgc"]) if opp is not None
                            else float(pitch[team]["w"].sum(axis=0).mean()))
            scored[team] = rng.poisson(lam_of[team], size=n_sims)
        for fx in sides:
            team = int(fx["team_id"])
            rows = np.flatnonzero(team_of == team)
            if len(rows) == 0:
                continue
            opp = next((o for o in sides if int(o["team_id"]) != team), None)
            conceded = scored[int(opp["team_id"])] if opp is not None \
                else rng.poisson(float(fx["xgc"]), size=n_sims)
            pts, goals = _score_side(R[rows], M[rows], positions[rows], fx, pitch[team],
                                     scored[team], conceded, lam_of[team], n_sims, rng)
            samples[rows] += pts
            # In a double gameweek one appearance is enough to keep the
            # captain's armband and prevent an autosub, so aggregate with OR.
            played_all[rows] |= pitch[team]["played"]
            goals_all[rows] += goals
            conceded_all[rows] += conceded[None, :]
    return ids, samples, {"played": played_all, "goals": goals_all,
                          "conceded_team": conceded_all, "team_of": team_of}


def _on_pitch(R, M, rows, fx, n_sims, rng) -> dict:
    """Minutes on the pitch per player-scenario, and each one's goal weight.

    `p_start` is drawn ONCE per player as a plain Bernoulli, not per-scenario
    from a posterior. An earlier version drew a fresh Beta `p_start` per
    scenario, reasoning that a price prior and an ever-present's thirty
    starts can land on the same 0.8 and should not be simulated with equal
    confidence. That reasoning is correct, but the mechanism could not
    deliver it: for a SINGLE binary event in a SINGLE gameweek, the marginal
    distribution of "did he start" is Bernoulli(p_start) for ANY generating
    process with that mean -- there is no way to "widen" a hard 0/1 outcome
    beyond the variance its own mean already fixes (p*(1-p)), so drawing a
    per-scenario theta changed nothing real. Worse, computing the conditional
    60-minute probability as `p_60 / theta` (theta, not the fixed p_start) was
    an active bug: since `theta * (p_60/theta) = p_60` exactly whenever
    unclipped, but the ratio gets clipped to 1 whenever theta dips below
    p_60, the clip fires disproportionately in exactly the low-theta
    scenarios and pulls the true `P(reached 60)` below the intended `p_60` --
    confirmed numerically (0.402 simulated against an intended 0.45 at
    moderate evidence). Genuine epistemic uncertainty about a role would need
    a THETA REUSED ACROSS A MULTI-WEEK DECISION -- the same player is more or
    less nailed on in every week of a transfer's horizon, not independently
    per gameweek -- which is a different, larger piece of work than this
    single-event simulator does (see R4/R8 in the audit). `minutes.py` still
    computes and exposes `start_evidence` as a labelled confidence signal for
    that future extension; nothing in this module reads it.
    """
    p_start, p_play, p_60, m_start = (M[rows, i][:, None] for i in range(4))
    n = len(rows)
    u = rng.random((n, n_sims))
    started = u < p_start
    subbed = (u >= p_start) & (u < p_play)
    mins = np.where(started, m_start, np.where(subbed, M_SUB, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        p60_given_start = np.where(p_start > 0, p_60 / np.maximum(p_start, 1e-12), 0.0)
    reached_60 = started & (rng.random((n, n_sims)) < np.clip(p60_given_start, 0.0, 1.0))
    share = mins / 90.0
    att = float(fx["att_mult"])
    # Each player's expected goals in this scenario, given his minutes. The
    # same quantity model.xp integrates; here it is the weight his side's
    # goals are allocated by.
    w = np.maximum(R[rows, 0][:, None] * share * att, 0.0)
    return {"played": started | subbed, "reached_60": reached_60, "share": share, "w": w}


def _allocate(total: np.ndarray, w: np.ndarray, cap: float, rng) -> np.ndarray:
    """Hand a side's goals to its players, preserving each player's mean.

    Goal by goal, each of the side's `total` goals goes to player i with
    probability w_i / D, where D = max(sum w, cap) per scenario. When the
    players' modelled total is within the side's expected total the shares
    sum to less than one and the remainder is a goal by nobody in the frame
    -- an own goal, a player with no minutes on record -- so every player's
    marginal stays exactly Poisson(w_i). When the players' total EXCEEDS the
    side's, shares are normalised and every attacker is scaled down to fit
    the side's total: the strength model's team number wins over the sum of
    individual rates, which is also the audit's R7 remedy for counting team
    quality twice. Drawn as successive conditional binomials, which is a
    multinomial without the per-scenario loop.
    """
    n, n_sims = w.shape
    W = w.sum(axis=0)
    D = np.maximum(W, cap)
    out = np.zeros((n, n_sims), dtype=float)
    remaining = total.astype(int).copy()
    remaining_p = np.ones(n_sims)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            p_i = np.where(D > 0, w[i] / D, 0.0)
            cond = np.where(remaining_p > 1e-12, np.clip(p_i / remaining_p, 0.0, 1.0), 0.0)
            g = rng.binomial(remaining, cond)
            out[i] = g
            remaining = remaining - g
            remaining_p = remaining_p - p_i
    return out


def _score_side(R, M, positions, fx, pitch, team_goals, conceded_team, side_lambda,
                n_sims, rng):
    """One side's points in one fixture, given the match's scoreline.

    `side_lambda` is the expected goals the side's total was drawn at, and is
    the cap the allocation preserves player means against.
    """
    xg90, xa90, bonus90, dc90, saves90, cards90 = (R[:, i][:, None] for i in range(6))
    played, reached_60, share = pitch["played"], pitch["reached_60"], pitch["share"]
    n = len(R)

    goals = _allocate(team_goals, pitch["w"], float(side_lambda), rng)

    # Goals conceded while a given player was on the pitch: thinning the side's
    # conceded count by time on pitch keeps his marginal tied to his teammates'.
    conceded_on = rng.binomial(np.broadcast_to(conceded_team[None, :], (n, n_sims)).astype(int),
                               np.clip(share, 0.0, 1.0))
    # A mean-1 team attacking multiplier for the parts still drawn per player.
    theta = rng.gamma(TEAM_FORM_SHAPE, 1.0 / TEAM_FORM_SHAPE, size=n_sims)[None, :]
    att = float(fx["att_mult"])
    # Assists stay independent of the allocated goals: a coherent assist model
    # (at most one per goal, allocated by xA share) would move assist means
    # away from model.xp unless the two were re-derived together. Recorded as
    # a residual; goals and clean sheets are the pair the rank layer turns on.
    assists = rng.poisson(np.maximum(xa90 * share * att * theta, 0.0))
    dc = rng.poisson(np.maximum(dc90 * share, 0.0))
    threat = float(fx["opp_threat"]) if "opp_threat" in fx else 1.0
    saves = rng.poisson(np.maximum(saves90 * share * threat, 0.0))
    cards = rng.poisson(np.maximum(cards90 * share, 0.0))
    scale = 1.0 + (att - 1.0) * FIXTURE_SENSITIVITY
    b = np.clip(bonus90 * share * scale, 0.0, MAX_BONUS_PER_MATCH)
    bonus = rng.binomial(int(MAX_BONUS_PER_MATCH), b / MAX_BONUS_PER_MATCH)

    goal_pts = np.array([GOAL_PTS[p] for p in positions])[:, None]
    cs_pts = np.array([CS_PTS[p] for p in positions])[:, None]
    dc_bar = np.array([DC_THRESHOLD.get(p, 12) for p in positions])[:, None]
    is_keeper = (positions == "GKP")[:, None]
    concedes = np.isin(positions, list(CONCEDED_PENALTY_POSITIONS))[:, None]

    pts = played * 1.0 + reached_60 * 1.0
    pts += goals * goal_pts + assists * ASSIST_PTS
    # The clean sheet is the SIDE's, derived from the same scoreline the
    # opposition's goals came from, and it needs the hour.
    pts += (conceded_team[None, :] == 0) * reached_60 * cs_pts
    pts += (dc >= dc_bar) * DC_PTS
    pts += bonus
    pts -= concedes * (conceded_on // CONCEDED_PER_PENALTY)
    pts += is_keeper * (saves // SAVES_PER_POINT)
    pts -= cards
    return pts.astype(float), goals


def moment_match(samples: np.ndarray, ids: list[int], xp: pd.DataFrame,
                 xp_col: str = "xp_next1") -> np.ndarray:
    """Rescale each player's samples so their mean equals his projected `xp_col`.

    The simulation draws from RAW rates and minutes. Once per-position
    calibration has moved the projection, the MILP is choosing on calibrated
    numbers while the rank layer scores candidates on samples whose means are
    the old ones -- so the squad that "beats the field most often" is judged
    on a distribution that does not describe it. Scaling each row by
    calibrated / simulated mean keeps the shape (zeros stay zeros, the tail
    stays a tail) and makes the two agree exactly on the mean. A player whose
    simulated mean is zero has nothing to scale and is left alone.
    """
    target = xp.set_index("player_id").reindex(ids)[xp_col].astype(float).to_numpy()
    means = samples.mean(axis=1)
    factor = np.ones_like(means)
    ok = (means > 1e-9) & np.isfinite(target)
    factor[ok] = target[ok] / means[ok]
    return samples * factor[:, None]
