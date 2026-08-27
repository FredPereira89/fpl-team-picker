"""Mode 2: weekly transfer optimization.

Search depth is free_transfers + max_paid_hits, never a fixed constant:
free transfers bank up to 5, and a fixed cap would silently discard moves
that cost nothing.
"""
from dataclasses import dataclass, field
import pandas as pd
import pulp

from .squad import SQUAD_SPLIT, XI_MIN, XI_MAX, XI_SIZE, MAX_PER_CLUB
import numpy as np


def selling_price(purchase: float, now: float) -> float:
    """What FPL actually pays for a player.

    A rise is only half yours, and only in whole 0.1 steps: bought at 5.0 and
    now worth 5.3, you sell for 5.1, not 5.3. A fall is taken in full. Working
    from `now_cost` alone -- which is what this module did until the 2026-08-27
    audit -- hands the solver a budget it cannot realise, and the error grows
    every week a squad is held.
    """
    purchase, now = float(purchase), float(now)
    if now <= purchase:
        return now
    rise_steps = round((now - purchase) * 10)
    return round(purchase + (rise_steps // 2) / 10.0, 1)


@dataclass
class TransferPlan:
    out_ids: list[int] = field(default_factory=list)
    in_ids: list[int] = field(default_factory=list)
    n_transfers: int = 0
    hit_cost: int = 0
    squad_ids: list[int] = field(default_factory=list)
    starting_ids: list[int] = field(default_factory=list)
    gross_xp: float = 0.0
    net_xp: float = 0.0
    baseline_xp: float = 0.0
    gain: float = 0.0


def _solve(xp_df, current, budget, max_changes, cfg, xp_col, cost=None):
    ids = [int(i) for i in xp_df["player_id"]]
    xp = dict(zip(ids, xp_df[xp_col].astype(float)))
    price = dict(zip(ids, xp_df["price"].astype(float)))
    cost = cost or price
    pos = dict(zip(ids, xp_df["position"]))
    club = dict(zip(ids, xp_df["team"]))
    bench_w = float(np.mean(cfg.bench_weight))
    current_set = set(int(i) for i in current)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    # Captaincy doubles one starter -- see optimize.squad for why it belongs in
    # the objective rather than being applied after the fact.
    cap = pulp.LpVariable.dicts("cap", ids, cat="Binary")
    prob += pulp.lpSum(
        xp[i] * start[i] + bench_w * xp[i] * (squad[i] - start[i]) + xp[i] * cap[i]
        for i in ids
    )
    prob += pulp.lpSum(cap[i] for i in ids) == 1
    for i in ids:
        prob += cap[i] <= start[i]
    prob += pulp.lpSum(cost.get(i, price[i]) * squad[i] for i in ids) <= budget
    prob += pulp.lpSum(squad[i] for i in ids) == sum(SQUAD_SPLIT.values())
    prob += pulp.lpSum(start[i] for i in ids) == XI_SIZE
    for p, n in SQUAD_SPLIT.items():
        prob += pulp.lpSum(squad[i] for i in ids if pos[i] == p) == n
        in_pos = [start[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(in_pos) >= XI_MIN[p]
        prob += pulp.lpSum(in_pos) <= XI_MAX[p]
    for c in set(club.values()):
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= MAX_PER_CLUB
    for i in ids:
        prob += start[i] <= squad[i]
    # keep at least 15 - max_changes of the current squad
    prob += pulp.lpSum(squad[i] for i in ids if i in current_set) >= 15 - max_changes

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
        return None
    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    captain = next((i for i in ids if cap[i].value() > 0.5), None)
    return chosen, starters, sum(xp[i] for i in starters) + (xp[captain] if captain else 0.0)


def optimize_transfers(xp_df: pd.DataFrame, current_squad_ids: list[int], bank: float,
                       free_transfers: int, cfg, xp_col: str = "xp_next5",
                       selling_prices: dict[int, float] | None = None):
    """Best transfer plan for the week.

    `selling_prices` is what FPL would pay for each player already owned; it
    defaults to market price, which is correct only for a squad that has not
    moved in price. Charging owned players at their SELLING value on both sides
    of the budget constraint keeps holding a risen player free while making the
    proceeds from selling him honest.
    """
    current_set = set(int(i) for i in current_squad_ids)
    price = dict(zip(xp_df["player_id"].astype(int), xp_df["price"].astype(float)))
    cost = dict(price)
    for pid in current_set:
        cost[pid] = float((selling_prices or {}).get(pid, price[pid]))
    budget = float(bank) + sum(cost[i] for i in current_set)

    options: list[TransferPlan] = []
    for n in range(0, int(free_transfers) + int(cfg.max_paid_hits) + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost)
        if solved is None:
            continue
        chosen, starters, gross = solved
        actual = len(current_set - set(chosen))
        paid = max(0, actual - int(free_transfers))
        hit = paid * int(cfg.hit_cost)
        options.append(TransferPlan(
            out_ids=sorted(current_set - set(chosen)),
            in_ids=sorted(set(chosen) - current_set),
            n_transfers=actual,
            hit_cost=hit,
            squad_ids=chosen,
            starting_ids=starters,
            gross_xp=round(gross, 3),
            net_xp=round(gross - hit, 3),
        ))

    # options[0] is the 0-transfer baseline; always keep it visible
    baseline = options[0].net_xp if options else 0.0
    for o in options:
        o.baseline_xp = baseline
        o.gain = round(o.net_xp - baseline, 3)
    best = max(options, key=lambda o: o.net_xp)
    return best, options
