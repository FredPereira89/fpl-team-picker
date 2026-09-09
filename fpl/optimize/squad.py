"""Mode 1: build a full 15-man squad from scratch (initial team / wildcard / free hit).

Single joint MILP over squad and starting-XI membership. A two-stage
"pick 15 then pick 11" would spend budget on bench players who score nothing.
"""
from dataclasses import dataclass, field
import pandas as pd
import pulp

from .objective import (add_bench, add_captaincy, captain_bonus, captain_values,
                        chosen_captains, first_captain, tilted_frame)

SQUAD_SPLIT = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3


@dataclass
class Squad:
    player_ids: list[int]
    starting_ids: list[int]
    total_cost: float
    xp: float
    captain_id: int | None = None
    # Who wears the armband in each gameweek of the horizon. It is free to move
    # every week, so a single captain_id describes only the first of them.
    captains: dict = field(default_factory=dict)


def enumerate_squads(xp_df: pd.DataFrame, cfg, xp_col: str = "xp_next5", k: int = 8,
                     must_include: list[int] | None = None,
                     banned: list[int] | None = None,
                     min_different: int = 4) -> list[Squad]:
    """The solver's `k` best squads, best first.

    A MILP maximises a LINEAR objective, and rank is not linear in the squad --
    P(beat the field) depends on the joint distribution of fifteen correlated
    players, which no objective row can express. So the solver is used for what
    it is good at, proposing strong squads, and `optimize.rank` chooses between
    them by simulation.

    Successive solutions are forced apart by a no-good cut: having returned a
    squad, require at least `min_different` of its fifteen to be dropped.

    `min_different` is the setting that decides whether any of this is worth
    doing. At 1 the candidates are the same team with one player swapped, and
    on real GW4 data rank selection then re-picked the plain optimum at every
    target -- seven seconds of search that changed nothing. At 4 the candidates
    are genuinely different teams, and the choice starts to matter: the same
    data gave up 0.5 xP to move P(a top-tenth week) from 0.593 to 0.651.

    Returns fewer than `k` if the pool runs out of squads that far apart.
    """
    out: list[Squad] = []
    excluded: list[list[int]] = []
    for _ in range(int(k)):
        squad = _solve_squad(xp_df, cfg, xp_col, must_include, banned, excluded,
                             min_different=int(min_different))
        if squad is None:
            break
        out.append(squad)
        excluded.append(list(squad.player_ids))
    return out


def optimize_squad(xp_df: pd.DataFrame, cfg, xp_col: str = "xp_next5",
                   must_include: list[int] | None = None,
                   banned: list[int] | None = None) -> Squad:
    """The single best squad by expected points."""
    return _solve_squad(xp_df, cfg, xp_col, must_include, banned, excluded=None)


def _solve_squad(xp_df, cfg, xp_col, must_include, banned, excluded,
                 min_different: int = 1):
    pool = xp_df[~xp_df["player_id"].isin(banned or [])].reset_index(drop=True)
    ids = [int(i) for i in pool["player_id"]]
    # The solver optimises the ownership-tilted score; everything reported back
    # is read from `pool`, so the user still sees real expected points.
    tilted = tilted_frame(pool, cfg, xp_col)
    xp = dict(zip(ids, tilted[xp_col].astype(float)))
    raw_xp = dict(zip(ids, pool[xp_col].astype(float)))
    price = dict(zip(ids, pool["price"].astype(float)))
    pos = dict(zip(ids, pool["position"]))
    club = dict(zip(ids, pool["team"]))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    # The captain scores twice. Leaving this out of the objective made the
    # solver indifferent between a squad with one high ceiling and a squad of
    # equal total spread flat -- and captaincy is roughly a sixth of a
    # gameweek's score. The armband is re-chosen every gameweek, so it is
    # valued per gameweek whenever the frame carries a per-gameweek breakdown.
    cap_terms, cap_vars, cap_values = add_captaincy(prob, ids, tilted, start, cfg, xp_col)
    bench_terms = add_bench(prob, ids, xp, pos, squad, start, cfg)

    prob += pulp.lpSum(xp[i] * start[i] for i in ids) + bench_terms + cap_terms

    prob += pulp.lpSum(price[i] * squad[i] for i in ids) <= cfg.budget
    prob += pulp.lpSum(squad[i] for i in ids) == sum(SQUAD_SPLIT.values())
    prob += pulp.lpSum(start[i] for i in ids) == XI_SIZE
    for p, n in SQUAD_SPLIT.items():
        prob += pulp.lpSum(squad[i] for i in ids if pos[i] == p) == n
    for p in SQUAD_SPLIT:
        in_pos = [start[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(in_pos) >= XI_MIN[p]
        prob += pulp.lpSum(in_pos) <= XI_MAX[p]
    for c in set(club.values()):
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= MAX_PER_CLUB
    for i in ids:
        prob += start[i] <= squad[i]
    for i in (must_include or []):
        if i not in squad:
            raise ValueError(f"must_include player {i} is not in the pool")
        prob += squad[i] == 1

    for combo in (excluded or []):
        present = [i for i in combo if i in squad]
        if present:
            drop = min(int(min_different), len(present))
            prob += pulp.lpSum(squad[i] for i in present) <= len(present) - drop

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        if excluded:
            return None          # the pool ran out of distinct squads
        raise ValueError(
            f"squad selection infeasible under the given constraints "
            f"(budget £{cfg.budget}m, status={pulp.LpStatus[status]})"
        )

    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    # Reported in UNTILTED points: the tilt is a solver preference, not a
    # forecast, so what a human is shown stays real expected points. The
    # armband is still valued week by week and discounted, exactly as the
    # solver valued it -- only the ownership tilt is stripped back out.
    captain = first_captain(cap_vars)
    total = sum(raw_xp[i] for i in starters) + captain_bonus(
        cap_vars, captain_values(pool, ids, cfg, xp_col))
    return Squad(
        player_ids=chosen,
        starting_ids=starters,
        total_cost=round(sum(price[i] for i in chosen), 1),
        xp=round(total, 3),
        captain_id=captain,
        captains=chosen_captains(cap_vars),
    )
