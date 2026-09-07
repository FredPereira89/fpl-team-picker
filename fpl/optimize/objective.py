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
import pulp

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
