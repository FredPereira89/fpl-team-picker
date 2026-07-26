"""Mode 1: build a full 15-man squad from scratch (initial team / wildcard / free hit).

Single joint MILP over squad and starting-XI membership. A two-stage
"pick 15 then pick 11" would spend budget on bench players who score nothing.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pulp

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


def optimize_squad(xp_df: pd.DataFrame, cfg, xp_col: str = "xp_next5",
                   must_include: list[int] | None = None,
                   banned: list[int] | None = None) -> Squad:
    pool = xp_df[~xp_df["player_id"].isin(banned or [])].reset_index(drop=True)
    ids = [int(i) for i in pool["player_id"]]
    xp = dict(zip(ids, pool[xp_col].astype(float)))
    price = dict(zip(ids, pool["price"].astype(float)))
    pos = dict(zip(ids, pool["position"]))
    club = dict(zip(ids, pool["team"]))
    bench_w = float(np.mean(cfg.bench_weight))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")

    prob += pulp.lpSum(
        xp[i] * start[i] + bench_w * xp[i] * (squad[i] - start[i]) for i in ids
    )

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

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"squad selection infeasible under the given constraints "
            f"(budget £{cfg.budget}m, status={pulp.LpStatus[status]})"
        )

    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    return Squad(
        player_ids=chosen,
        starting_ids=starters,
        total_cost=round(sum(price[i] for i in chosen), 1),
        xp=round(sum(xp[i] for i in starters), 3),
    )
