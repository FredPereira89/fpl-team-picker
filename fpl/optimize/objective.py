"""Objective pieces shared by the squad and transfer solvers.

Both were valuing two things wrongly in the same way, so both are fixed here
rather than twice:

* the BENCH was priced at the mean of the four configured slot weights, which
  makes Bench 3 as valuable as Bench 1 and quietly spends budget on a player
  who comes on about one week in fifty;
* the ARMBAND was fixed for the whole horizon, so a one-week fixture spike was
  valued as though it had to be captained for five gameweeks -- when in reality
  captaincy is re-chosen, free, every week.
"""
import pandas as pd
import pulp

# Ownership tilt. `differential` rewards players the field does not own;
# `template` rewards matching it. Both are the same axis with opposite sign.
PROFILE_SIGN = {"balanced": 0.0, "differential": 1.0, "template": -1.0}
# The most the tilt may move a projection, as a fraction of it, at weight 1.0.
# Deliberately small -- see effective_xp for why this is a tie-breaker and not
# a rank model.
MAX_TILT = 0.15


def effective_xp(xp_df, cfg, xp_col: str):
    """Expected points, nudged by how much of the field already owns the player.

    Worth being precise about what ownership can and cannot buy, because the
    obvious formula is wrong. If you own player i you score `xp_i` and the
    average rival scores `EO_i * xp_i`, so your edge is `xp_i * (1 - EO_i)`.
    But if you DON'T own him your edge is `-EO_i * xp_i` -- and the difference
    between owning and not owning is `xp_i` either way. Effective ownership
    cancels out of EXPECTED rank entirely. What it actually changes is the
    VARIANCE of your rank: the template holds your position, a differential
    widens the distribution in both directions.

    Ranking on `xp * (1 - EO)` therefore does not maximise expected rank, it
    just punts -- at weight 0.5 it preferred a player projected two points
    lower purely for being unowned. Modelling this properly needs a
    distribution over each player's points and over the field's, which this
    model does not have.

    So the tilt here is bounded to +/-`MAX_TILT` of a projection and is honestly
    only a tie-breaker: among options the model cannot separate on points,
    lean differential (chasing rank) or template (defending it). At
    `ownership_weight = 0`, the default, this returns `xp_col` untouched.
    """
    xp = xp_df[xp_col].astype(float)
    weight = float(getattr(cfg, "ownership_weight", 0.0) or 0.0)
    sign = PROFILE_SIGN.get(getattr(cfg, "risk_profile", "balanced"), 0.0)
    if weight == 0.0 or sign == 0.0 or "ownership" not in xp_df.columns:
        return xp
    return xp * tilt_factor(xp_df, cfg)


def tilt_factor(xp_df, cfg):
    """Per-player multiplier the ownership tilt applies. 1.0 everywhere when off.

    Returned separately so a solver can scale EVERY objective column by the
    same factor -- the horizon total and each per-gameweek column that carries
    the armband -- instead of tilting the base term and leaving captaincy
    untilted, which would rank a player differently depending on which term
    was looking at him.
    """
    weight = float(getattr(cfg, "ownership_weight", 0.0) or 0.0)
    sign = PROFILE_SIGN.get(getattr(cfg, "risk_profile", "balanced"), 0.0)
    if weight == 0.0 or sign == 0.0 or "ownership" not in xp_df.columns:
        return pd.Series(1.0, index=xp_df.index)
    share = xp_df["ownership"].astype(float).clip(lower=0.0, upper=100.0) / 100.0
    # (1 - 2*share) runs +1 for a wholly unowned player to -1 for a universal one.
    return 1.0 + weight * MAX_TILT * sign * (1.0 - 2.0 * share)


def tilted_frame(xp_df, cfg, xp_col: str):
    """A copy of the frame with every objective column scaled by the tilt.

    The solver maximises this; the caller reports from the ORIGINAL frame, so
    what a user is shown stays honest expected points -- the same split as
    `xp_horizon` (what the solver maximises) beside `xp_next5` (what is shown).
    """
    factor = tilt_factor(xp_df, cfg)
    if (factor == 1.0).all():
        return xp_df
    out = xp_df.copy()
    for col in [xp_col] + [c for _, c in event_columns(xp_df)]:
        if col in out.columns:
            out[col] = out[col].astype(float) * factor
    return out


# Per-gameweek expected points, written by model.xp as xp_gw{event}.
EVENT_PREFIX = "xp_gw"
# A captain is never outside the top of the projection, and every candidate
# adds a column per gameweek to the LP. 60 is far past any plausible armband.
CAPTAIN_CANDIDATES = 60


def event_columns(xp_df) -> list[tuple[int, str]]:
    """(event, column) for every per-gameweek xP column, earliest first."""
    out = []
    for c in xp_df.columns:
        name = str(c)
        if name.startswith(EVENT_PREFIX) and name[len(EVENT_PREFIX):].isdigit():
            out.append((int(name[len(EVENT_PREFIX):]), name))
    return sorted(out)


def captain_values(xp_df, ids, cfg, xp_col) -> dict:
    """{event: {player: value of the armband in that event}}.

    Values are discounted the same way `xp_horizon` is, so the captain term and
    the base term are in the same currency. A frame with no per-gameweek
    columns falls back to a single captain over the whole horizon, which is
    what every caller got before this existed.
    """
    frame = xp_df.set_index("player_id")
    decay = float(getattr(cfg, "horizon_decay", 1.0))
    cols = event_columns(xp_df)
    if not cols:
        return {None: {i: float(frame.loc[i, xp_col]) for i in ids}}
    return {
        event: {i: float(frame.loc[i, col]) * decay ** offset for i in ids}
        for offset, (event, col) in enumerate(cols)
    }


def add_captaincy(prob, ids, xp_df, start, cfg, xp_col):
    """Add the captaincy variables and return (objective terms, variables).

    One captain per gameweek, each constrained to a starter. The variables are
    continuous rather than binary on purpose: for any fixed starting XI the
    choice is "put the armband on the best available starter", whose LP optimum
    is already a vertex -- so this costs the solver nothing in integrality and
    saves it thousands of branch-and-bound columns.

    The constraint is `<= 1`, never `== 1`: candidates are capped at the top of
    the projection, and a squad containing none of them would make an equality
    infeasible rather than merely unattractive.
    """
    values = captain_values(xp_df, ids, cfg, xp_col)
    terms, cap_vars = [], {}
    for event, vals in values.items():
        pool = sorted(ids, key=lambda i: vals[i], reverse=True)[:CAPTAIN_CANDIDATES]
        tag = "h" if event is None else str(event)
        v = pulp.LpVariable.dicts(f"cap_{tag}", pool, lowBound=0, upBound=1)
        prob += pulp.lpSum(v[i] for i in pool) <= 1
        for i in pool:
            prob += v[i] <= start[i]
        terms.append(pulp.lpSum(vals[i] * v[i] for i in pool))
        cap_vars[event] = v
    return pulp.lpSum(terms), cap_vars, values


def chosen_captains(cap_vars) -> dict:
    """{event: player_id} actually selected, skipping events with no captain."""
    out = {}
    for event, v in cap_vars.items():
        picked = [i for i, var in v.items() if (var.value() or 0) > 0.5]
        if picked:
            out[event] = int(picked[0])
    return out


def captain_bonus(cap_vars, values) -> float:
    """Points the armband is worth in the solved plan."""
    return sum(values[event][pid] for event, pid in chosen_captains(cap_vars).items())


def first_captain(cap_vars):
    """This week's armband: the captain for the earliest gameweek in the plan."""
    chosen = chosen_captains(cap_vars)
    if not chosen:
        return None
    dated = {e: p for e, p in chosen.items() if e is not None}
    return chosen[min(dated)] if dated else next(iter(chosen.values()))


def add_bench(prob, ids, xp, pos, squad, start, cfg):
    """Add ordered bench slots and return their objective terms.

    FPL substitutes in bench order, so Bench 1 is worth several times Bench 3.
    Averaging the configured weights -- which both solvers did -- prices them
    identically and buys a £4.5m fourth-choice player instead of upgrading the
    XI.

    `cfg.bench_weight` is [bench 1, bench 2, bench 3, reserve keeper]: the
    three outfield slots in substitution order, then the second goalkeeper, who
    plays only when the first does not and is worth the least of the four.

    Like the captaincy variables these are continuous. For a fixed squad and XI
    the slot assignment is a transportation problem, whose LP relaxation is
    integral, so nothing is lost by not branching on them.
    """
    w_outfield = [float(w) for w in cfg.bench_weight[:3]]
    w_keeper = float(cfg.bench_weight[3])
    outfield = [i for i in ids if pos[i] != "GKP"]
    keepers = [i for i in ids if pos[i] == "GKP"]

    slots = {
        s: pulp.LpVariable.dicts(f"bench{s}", outfield, lowBound=0, upBound=1)
        for s in range(len(w_outfield))
    }
    for s in slots:
        prob += pulp.lpSum(slots[s][i] for i in outfield) == 1
    for i in outfield:
        prob += pulp.lpSum(slots[s][i] for s in slots) == squad[i] - start[i]

    return pulp.lpSum(
        [w_outfield[s] * xp[i] * slots[s][i] for s in slots for i in outfield]
        + [w_keeper * xp[i] * (squad[i] - start[i]) for i in keepers]
    )
