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
from collections import deque

import numpy as np
import pandas as pd

from .minutes import M_START, M_SUB
from .xp import (GOAL_PTS, CS_PTS, ASSIST_PTS, DC_PTS, DC_THRESHOLD,
                 SAVES_PER_POINT, CONCEDED_PER_PENALTY, CONCEDED_PENALTY_POSITIONS,
                 feasible_goal_and_assist_marginals)
from .bps import score_side_bps, award_match_bonus

DEFAULT_SIMS = 4000
STARTERS_PER_TEAM = 11
MAX_SUBSTITUTES_PER_TEAM = 5
# A transport residual of 1e-10 is many orders below Monte Carlo error even
# for a million draws; never silently sample from an unfinished coupling.
TRANSPORT_TOLERANCE = 1e-10
# IPF is fast for well-conditioned cases.  Difficult sparse transports route
# to the exact max-flow fallback below rather than spending thousands of
# vectorised passes asymptotically approaching the answer.
TRANSPORT_MAX_ITERATIONS = 200


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

    {"ids", "samples", "played", "started", "subbed", "goals", "assists",
    "conceded_team", "conceded_on", "team_goals", "team_of", "bonus"} --
    `played` records whether each player appeared in
    at least one fixture in the gameweek.  Keeping appearance separate from
    points matters because a player can appear and score zero: captaincy
    passes to the vice only on NO appearance, and autosubs use the same
    rule.  `goals` is each player's goals per scenario and `conceded_team`
    the goals his SIDE conceded in that scenario, so the coherence property
    (no goal against a clean sheet in the same match) can be asserted
    directly rather than inferred. `conceded_on` is goals conceded while
    THIS PLAYER was on the pitch specifically (shared goal timing across
    teammates -- see `_conceded_on` -- so two players with the same window
    always agree), exposed so a caller can assert the clean-sheet EVENT
    (`conceded_on == 0`) directly instead of reverse-engineering it out of
    `samples`, which also carries goals/assists/DC/cards/bonus. `bonus` is
    each player's own share of the match-wide 3/2/1 ranking (see
    `model.bps`), exposed separately from `samples` so a caller can compare
    the OTHER scoring components against the analytic projection without
    the bonus mechanism's own mean (which is a ranked, scarce match
    resource, not an independent per-player rate) obscuring the comparison.
    """
    ids, samples, detail = _simulate(players, rates, minutes, tfx, event, n_sims, seed)
    return {"ids": ids, "samples": samples, **detail}


def _simulate(players, rates, minutes, tfx, event, n_sims, seed):
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in players["player_id"]]
    samples = np.zeros((len(ids), n_sims), dtype=float)
    played_all = np.zeros((len(ids), n_sims), dtype=bool)
    started_all = np.zeros((len(ids), n_sims), dtype=bool)
    subbed_all = np.zeros((len(ids), n_sims), dtype=bool)
    goals_all = np.zeros((len(ids), n_sims), dtype=float)
    assists_all = np.zeros((len(ids), n_sims), dtype=float)
    conceded_all = np.zeros((len(ids), n_sims), dtype=float)
    bonus_all = np.zeros((len(ids), n_sims), dtype=float)
    conceded_on_all = np.zeros((len(ids), n_sims), dtype=float)

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
    team_goals_all: dict[int, np.ndarray] = {}
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
            team_goals_all[team] = team_goals_all.get(team, np.zeros(n_sims, dtype=int)) + scored[team]
        # Bonus is a MATCH-wide ranking (the three best BPS across BOTH sides),
        # not a per-side one, so every side's BPS is collected before any of
        # them is awarded -- see the loop below.
        match_rows, match_bps = [], []
        for fx in sides:
            team = int(fx["team_id"])
            rows = np.flatnonzero(team_of == team)
            if len(rows) == 0:
                continue
            opp = next((o for o in sides if int(o["team_id"]) != team), None)
            conceded = scored[int(opp["team_id"])] if opp is not None \
                else rng.poisson(float(fx["xgc"]), size=n_sims)
            pts, goals, assists, bps, conceded_on = _score_side(
                R[rows], M[rows], positions[rows], fx, pitch[team],
                scored[team], conceded, lam_of[team], n_sims, rng)
            samples[rows] += pts
            # In a double gameweek one appearance is enough to keep the
            # captain's armband and prevent an autosub, so aggregate with OR.
            played_all[rows] |= pitch[team]["played"]
            started_all[rows] |= pitch[team]["started"]
            subbed_all[rows] |= pitch[team]["subbed"]
            goals_all[rows] += goals
            assists_all[rows] += assists
            conceded_all[rows] += conceded[None, :]
            # Sums across a double gameweek's fixtures, same reasoning as
            # `conceded_all`: goals conceded while on the pitch, across
            # however many fixtures this event actually has.
            conceded_on_all[rows] += conceded_on
            match_rows.append(rows)
            match_bps.append(bps)
        if match_rows:
            bonus = award_match_bonus(np.concatenate(match_bps, axis=0))
            offset = 0
            for rows in match_rows:
                samples[rows] += bonus[offset:offset + len(rows)]
                bonus_all[rows] += bonus[offset:offset + len(rows)]
                offset += len(rows)
    return ids, samples, {"played": played_all, "started": started_all,
                          "subbed": subbed_all, "goals": goals_all,
                          "assists": assists_all, "conceded_team": conceded_all,
                          "team_goals": team_goals_all, "team_of": team_of,
                          "bonus": bonus_all, "conceded_on": conceded_on_all}


def _sample_fixed_slots(probabilities: np.ndarray, slots: int,
                        n_sims: int, rng) -> np.ndarray:
    """Sample fixed-size selections with the supplied inclusion marginals.

    Systematic sampling chooses exactly ``slots`` entries, including synthetic
    unmodelled squad places.  If probabilities sum to no more than the number
    of slots, every real player's inclusion probability is retained exactly;
    dummies absorb the residual.  This is the right dependency for a football
    lineup: an eleventh start makes a twelfth impossible, unlike independent
    Bernoulli draws.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if slots <= 0 or len(p) == 0:
        return np.zeros((len(p), n_sims), dtype=bool)
    if p.ndim == 1:
        p = np.broadcast_to(p[:, None], (len(p), n_sims)).copy()
    elif p.shape[1] != n_sims:
        raise ValueError("probability matrix must have one column per simulation")
    total = p.sum(axis=0)
    overfull = total > slots + 1e-9
    if overfull.any():
        # Hand-built frames can bypass minutes reconciliation.  Preserve their
        # relative ordering while refusing an impossible expected lineup.
        p *= np.minimum(1.0, float(slots) / np.maximum(total, 1e-12))[None, :]
        total = p.sum(axis=0)
    # Split unused places evenly across `slots` synthetic players.  This
    # vectorises the draw even when conditional substitute probabilities vary
    # by scenario, while each dummy remains a valid probability in [0, 1].
    dummies = np.broadcast_to(((float(slots) - total) / float(slots))[None, :],
                              (slots, n_sims))
    all_p = np.concatenate([p, dummies], axis=0)
    cumulative = np.cumsum(all_p, axis=0)
    targets = rng.random(n_sims)[None, :] + np.arange(slots)[:, None]
    picked = (cumulative[None, :, :] > targets[:, None, :]).argmax(axis=1)
    out = np.zeros((len(p), n_sims), dtype=bool)
    scenario = np.broadcast_to(np.arange(n_sims), picked.shape)
    real = picked < len(p)
    out[picked[real], scenario[real]] = True
    return out


def _on_pitch(R, M, rows, fx, n_sims, rng) -> dict:
    """Minutes on the pitch per player-scenario, and each one's goal weight.

    `p_start` is sampled as one coherent eleven, not as independent Bernoulli
    draws. An earlier version drew a fresh Beta `p_start` per
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
    started = _sample_fixed_slots(p_start[:, 0], STARTERS_PER_TEAM, n_sims, rng)
    # Conditional on not starting, each player retains the p_play marginal.
    # The fixed five-place draw makes substitute appearances mutually
    # exclusive too; dummies represent unused substitutions.
    with np.errstate(divide="ignore", invalid="ignore"):
        sub_given_not_start = np.where(
            p_start[:, 0] < 1.0 - 1e-12,
            (p_play[:, 0] - p_start[:, 0]) / np.maximum(1.0 - p_start[:, 0], 1e-12),
            0.0,
        )
    sub_given_not_start = np.clip(sub_given_not_start, 0.0, 1.0)
    q = np.where(started, 0.0, sub_given_not_start[:, None])
    subbed = _sample_fixed_slots(q, MAX_SUBSTITUTES_PER_TEAM, n_sims, rng)
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
    return {"played": started | subbed, "started": started, "subbed": subbed,
           "reached_60": reached_60, "share": share, "w": w}


def _joint_transport(scorer_marginal: np.ndarray,
                     assister_marginal: np.ndarray) -> np.ndarray:
    """Couple scorer/assister marginals with a zero real-player diagonal.

    Iterative proportional fitting is stopped by its *measured* largest row or
    column residual, not an arbitrary pass count.  A non-convergent transport
    is an invalid probability model, so it fails loudly rather than quietly
    biasing the weekly rank simulation.
    """
    categories, n_sims = scorer_marginal.shape
    joint = scorer_marginal[:, None, :] * assister_marginal[None, :, :]
    diagonal = np.arange(categories - 1)  # final row/column are the dummies
    joint[diagonal, diagonal, :] = 0.0

    for _ in range(TRANSPORT_MAX_ITERATIONS):
        row_sum = joint.sum(axis=1)
        joint *= np.divide(scorer_marginal, row_sum,
                           out=np.zeros_like(row_sum), where=row_sum > 1e-15)[:, None, :]
        col_sum = joint.sum(axis=0)
        joint *= np.divide(assister_marginal, col_sum,
                           out=np.zeros_like(col_sum), where=col_sum > 1e-15)[None, :, :]
        residual = max(
            float(np.max(np.abs(joint.sum(axis=1) - scorer_marginal))),
            float(np.max(np.abs(joint.sum(axis=0) - assister_marginal))),
        )
        if residual <= TRANSPORT_TOLERANCE:
            return joint
    # IPF can converge arbitrarily slowly when a zero-diagonal transport has
    # a nearly forced edge.  Solve only those unfinished scenario patterns as
    # an exact bounded network flow, then verify the same tolerance.
    row_residual = np.max(np.abs(joint.sum(axis=1) - scorer_marginal), axis=0)
    col_residual = np.max(np.abs(joint.sum(axis=0) - assister_marginal), axis=0)
    unfinished = np.flatnonzero(np.maximum(row_residual, col_residual) > TRANSPORT_TOLERANCE)
    if len(unfinished):
        signature = np.concatenate([
            scorer_marginal[:, unfinished].T,
            assister_marginal[:, unfinished].T,
        ], axis=1)
        _, inverse = np.unique(signature, axis=0, return_inverse=True)
        for group in range(int(inverse.max()) + 1):
            members = unfinished[inverse == group]
            exact = _exact_transport(scorer_marginal[:, members[0]],
                                     assister_marginal[:, members[0]])
            joint[:, :, members] = exact[:, :, None]

    final_residual = max(
        float(np.max(np.abs(joint.sum(axis=1) - scorer_marginal))),
        float(np.max(np.abs(joint.sum(axis=0) - assister_marginal))),
    )
    if final_residual > TRANSPORT_TOLERANCE:
        raise RuntimeError(
            "joint scorer/assister transport could not satisfy its marginals: "
            f"max residual {final_residual:.3e}"
        )
    return joint


def _exact_transport(scorer_marginal: np.ndarray,
                     assister_marginal: np.ndarray) -> np.ndarray:
    """Exact max-flow fallback for one zero-diagonal transport pattern."""
    categories = len(scorer_marginal)
    source, row0, col0, sink = 0, 1, 1 + categories, 1 + 2 * categories
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity):
        graph[start].append([end, len(graph[end]), float(capacity)])
        graph[end].append([start, len(graph[start]) - 1, 0.0])
        return len(graph[start]) - 1

    for i, value in enumerate(scorer_marginal):
        add_edge(source, row0 + i, value)
    references = {}
    for i in range(categories):
        for j in range(categories):
            if i == j and i != categories - 1:
                continue
            references[(i, j)] = add_edge(row0 + i, col0 + j, 1.0)
    for j, value in enumerate(assister_marginal):
        add_edge(col0 + j, sink, value)

    flow = 0.0
    total = float(scorer_marginal.sum())
    while flow < total - TRANSPORT_TOLERANCE:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for end, _, capacity in graph[node]:
                if capacity > 1e-15 and level[end] < 0:
                    level[end] = level[node] + 1
                    queue.append(end)
        if level[sink] < 0:
            break
        cursor = [0] * len(graph)

        def push(node, available):
            if node == sink:
                return available
            while cursor[node] < len(graph[node]):
                edge_index = cursor[node]
                end, reverse, capacity = graph[node][edge_index]
                if capacity > 1e-15 and level[end] == level[node] + 1:
                    sent = push(end, min(available, capacity))
                    if sent > 1e-15:
                        graph[node][edge_index][2] -= sent
                        graph[end][reverse][2] += sent
                        return sent
                cursor[node] += 1
            return 0.0

        while True:
            sent = push(source, total - flow)
            if sent <= 1e-15:
                break
            flow += sent

    if total - flow > TRANSPORT_TOLERANCE:
        raise RuntimeError(
            "joint scorer/assister transport is infeasible after applying "
            f"self-assist constraints (unrouted mass {total - flow:.3e})"
        )
    out = np.zeros((categories, categories), dtype=float)
    for (i, j), edge_index in references.items():
        # Reverse capacity equals the flow sent along this forward edge.
        end, reverse, _ = graph[row0 + i][edge_index]
        out[i, j] = graph[end][reverse][2]
    return out


def _allocate_goal_events(team_goals: np.ndarray, goal_weights: np.ndarray,
                          assist_weights: np.ndarray, side_lambda: float, rng):
    """Jointly allocate scorers and assisters one realised goal at a time.

    Every realised goal first receives its scorer.  The scorer is removed from
    the eligible assist pool for *that same goal*, which enforces FPL's final-
    pass definition while retaining both unassisted goals and goals/assists by
    unmodelled players.  A player may still score and assist in a multi-goal
    match, just never on the same individual goal.
    """
    n, n_sims = goal_weights.shape
    goals = np.zeros((n, n_sims), dtype=float)
    assists = np.zeros((n, n_sims), dtype=float)
    if side_lambda <= 0 or not np.any(team_goals):
        return goals, assists

    scorer_marginal, assister_marginal = feasible_goal_and_assist_marginals(
        goal_weights, assist_weights, side_lambda)

    # Couple the scorer and assister marginals as a transport problem.  The
    # extra row/column represent an unmodelled scorer and an unassisted goal.
    # Iterative proportional fitting preserves both marginal rates while the
    # zero diagonal forbids a player assisting their own goal.
    scorer_marginal = np.vstack([
        scorer_marginal,
        np.clip(1.0 - scorer_marginal.sum(axis=0), 0.0, 1.0),
    ])
    assister_marginal = np.vstack([
        assister_marginal,
        np.clip(1.0 - assister_marginal.sum(axis=0), 0.0, 1.0),
    ])
    categories = n + 1
    joint = _joint_transport(scorer_marginal, assister_marginal)
    cumulative = np.cumsum(joint.reshape(categories * categories, n_sims), axis=0)
    # Transport convergence ensures this alters only sub-1e-10 roundoff, while
    # making the categorical tail explicit and independent of platform float
    # summation order.
    cumulative[-1] = 1.0
    scenarios = np.arange(n_sims)
    for goal_number in range(int(np.max(team_goals))):
        active = team_goals > goal_number
        draw = rng.random(n_sims)
        selected = (cumulative > draw[None, :]).argmax(axis=0)
        scorer, assister = selected // categories, selected % categories
        real_scorer = active & (scorer < n)
        goals[scorer[real_scorer], scenarios[real_scorer]] += 1.0
        real_assister = active & (assister < n)
        assists[assister[real_assister], scenarios[real_assister]] += 1.0
    return goals, assists


def _conceded_on(conceded_team, started, subbed, share, n_sims, rng) -> np.ndarray:
    """Goals conceded while EACH player was on the pitch, shape (n, n_sims).

    Independently thinning each player's own count (`rng.binomial` per
    player-scenario cell) treats "was this goal scored while I was on" as
    an independent coin flip per player -- so two players with the IDENTICAL
    playing window can disagree about the SAME goal. Confirmed: two
    identical 60-minute players disagreed in 44.4% of scenarios (matching
    2*p*(1-p) at p=60/90 exactly), which breaks the very teammate
    correlation this simulator exists to capture.

    Fixed by giving the match's conceded goals a SHARED simulated timing --
    each drawn once per scenario as a fraction of the match elapsed,
    common to every player on this side -- and testing membership in each
    player's own interval: `[0, share]` if he started (on from kickoff
    until subbed off, or full time), `[1-share, 1]` if he came on as a
    substitute (on from when he entered until full time). Two players who
    share an interval now agree on every timed goal, and the timing
    assumption (uniform across the match) is a modelling simplification,
    not a source of the correlation bug.
    """
    # Time every goal that was actually drawn. A previous fixed cap of eight
    # silently changed a 10-goal scoreline into eight goals for the on-pitch
    # points/BPS calculations. The realised maximum is small for ordinary
    # Poisson football scores, so sizing this axis dynamically is both exact
    # and effectively the same cost in normal scenarios.
    max_goals = int(np.max(conceded_team)) if np.size(conceded_team) else 0
    if max_goals <= 0:
        return np.zeros(started.shape, dtype=int)

    g = np.arange(max_goals)[:, None, None]                           # (G,1,1)
    goal_active = g < conceded_team[None, None, :]                    # (G,1,S)
    goal_time = rng.random((max_goals, 1, n_sims))                    # (G,1,S), shared

    lo = np.where(started, 0.0, np.where(subbed, 1.0 - share, 0.0))[None, :, :]
    hi = np.where(started, share, np.where(subbed, 1.0, 0.0))[None, :, :]
    in_interval = (goal_time >= lo) & (goal_time <= hi) & goal_active  # (G,n,S)
    # A non-player has lo == hi == 0, which a continuous draw only matches
    # with probability zero -- true in principle, but an explicit mask
    # costs nothing and removes any doubt.
    played = (started | subbed)[None, :, :]
    return (in_interval & played).sum(axis=0)


def _score_side(R, M, positions, fx, pitch, team_goals, conceded_team, side_lambda,
                n_sims, rng):
    """One side's points in one fixture, given the match's scoreline.

    `side_lambda` is the expected goals the side's total was drawn at, and is
    the cap the allocation preserves player means against.
    """
    # bonus90 (column 2) is not used here -- bonus now comes from ranking
    # BPS across the match (see below), not from an independent per-player
    # rate.
    xg90, xa90, _, dc90, saves90, cards90 = (R[:, i][:, None] for i in range(6))
    played, reached_60, share = pitch["played"], pitch["reached_60"], pitch["share"]
    started, subbed = pitch["started"], pitch["subbed"]
    n = len(R)

    # Goals conceded while a given player was on the pitch, with SHARED
    # timing across every player on this side -- see `_conceded_on`.
    conceded_on = _conceded_on(conceded_team, started, subbed, share, n_sims, rng)
    att = float(fx["att_mult"])
    assist_weights = np.maximum(xa90 * share * att, 0.0)
    goals, assists = _allocate_goal_events(
        team_goals, pitch["w"], assist_weights, float(side_lambda), rng)
    dc = rng.poisson(np.maximum(dc90 * share, 0.0))
    threat = float(fx["opp_threat"]) if "opp_threat" in fx else 1.0
    saves = rng.poisson(np.maximum(saves90 * share * threat, 0.0))
    cards = rng.poisson(np.maximum(cards90 * share, 0.0))

    goal_pts = np.array([GOAL_PTS[p] for p in positions])[:, None]
    cs_pts = np.array([CS_PTS[p] for p in positions])[:, None]
    dc_bar = np.array([DC_THRESHOLD.get(p, 12) for p in positions])[:, None]
    is_keeper = (positions == "GKP")[:, None]
    concedes = np.isin(positions, list(CONCEDED_PENALTY_POSITIONS))[:, None]
    # PER-PLAYER: the official rule is no goal conceded WHILE ON THE PITCH,
    # not the match's final score -- a player subbed at 60' keeps his clean
    # sheet even if his side concedes afterwards. Using the SIDE's final
    # `conceded_team == 0` credited every player identically regardless of
    # substitution timing, denying it to anyone subbed before a later
    # concession. `conceded_on` is already the per-player, per-scenario
    # thinned count this needs.
    clean_sheet = (conceded_on == 0)

    pts = played * 1.0 + reached_60 * 1.0
    pts += goals * goal_pts + assists * ASSIST_PTS
    pts += clean_sheet * reached_60 * cs_pts
    pts += (dc >= dc_bar) * DC_PTS
    pts -= concedes * (conceded_on // CONCEDED_PER_PENALTY)
    pts += is_keeper * (saves // SAVES_PER_POINT)
    pts -= cards
    # Bonus is not awarded here: it is a MATCH-wide ranking of BPS across
    # both sides (see `_simulate`), not an independent per-player draw, so
    # this returns the ingredients for that ranking rather than a bonus
    # value of its own.
    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc, conceded_on)
    return pts.astype(float), goals, assists, bps, conceded_on


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
