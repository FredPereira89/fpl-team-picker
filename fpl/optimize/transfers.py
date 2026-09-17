"""Mode 2: weekly transfer optimization.

Search depth is free_transfers + max_paid_hits, never a fixed constant:
free transfers bank up to 5, and a fixed cap would silently discard moves
that cost nothing.
"""
from dataclasses import dataclass, field
import pandas as pd
import pulp

from .squad import SQUAD_SPLIT, XI_MIN, XI_MAX, XI_SIZE, MAX_PER_CLUB
from .squad import BENCH_FLOOR_COL
from .objective import (add_bench, add_captaincy, captain_bonus, captain_values,
                        tilted_frame)


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


def bank_after(bank: float, current_squad, new_squad, prices: dict[int, float],
               purchase_prices: dict[int, float] | None = None) -> float:
    """Cash left after swapping `current_squad` for `new_squad`.

    Proceeds are SELLING prices, not market prices: a player who has risen 0.3
    returns 0.1 of it. Retained players cancel out, so only the players who
    actually changed hands move money.
    """
    cur = {int(i) for i in current_squad}
    new = {int(i) for i in new_squad}
    sold, bought = cur - new, new - cur
    proceeds = sum(
        selling_price((purchase_prices or {}).get(i, prices[i]), prices[i]) for i in sold
    )
    return round(float(bank) + proceeds - sum(prices[i] for i in bought), 1)


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


def _solve(xp_df, current, budget, max_changes, cfg, xp_col, cost=None,
           excluded=None, min_different=1, bench_floor: float = 0.0,
           allow_club_overage: bool = False):
    ids = [int(i) for i in xp_df["player_id"]]
    # Solve on the tilted score, report the untilted one -- see optimize.squad.
    tilted = tilted_frame(xp_df, cfg, xp_col)
    xp = dict(zip(ids, tilted[xp_col].astype(float)))
    raw_xp = dict(zip(ids, xp_df[xp_col].astype(float)))
    price = dict(zip(ids, xp_df["price"].astype(float)))
    cost = cost or price
    pos = dict(zip(ids, xp_df["position"]))
    club = dict(zip(ids, xp_df["team"]))
    current_set = set(int(i) for i in current)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    # Captaincy doubles one starter, and the bench pays out in substitution
    # order -- see optimize.objective for why both belong in the objective
    # rather than being applied after the fact or averaged away.
    cap_terms, cap_vars, cap_values = add_captaincy(prob, ids, tilted, start, cfg, xp_col)
    bench_terms = add_bench(prob, ids, xp, pos, squad, start, cfg)
    prob += pulp.lpSum(xp[i] * start[i] for i in ids) + bench_terms + cap_terms
    prob += pulp.lpSum(cost.get(i, price[i]) * squad[i] for i in ids) <= budget
    prob += pulp.lpSum(squad[i] for i in ids) == sum(SQUAD_SPLIT.values())
    prob += pulp.lpSum(start[i] for i in ids) == XI_SIZE
    for p, n in SQUAD_SPLIT.items():
        prob += pulp.lpSum(squad[i] for i in ids if pos[i] == p) == n
        in_pos = [start[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(in_pos) >= XI_MIN[p]
        prob += pulp.lpSum(in_pos) <= XI_MAX[p]
    # A real Premier League transfer can temporarily leave an FPL manager with
    # four players from one club. FPL does NOT force a sale: the squad stands,
    # and must return to three only when the manager next makes a transfer. So
    # the zero-transfer hold is allowed to keep an existing overage -- forcing
    # the cap there made holding infeasible and manufactured a move the game
    # does not require. Every plan that does transfer, and every fresh or
    # chip-built squad, is capped at three.
    owned_per_club: dict = {}
    if allow_club_overage:
        for i in current_set:
            if i in club:
                owned_per_club[club[i]] = owned_per_club.get(club[i], 0) + 1
    for c in set(club.values()):
        cap = (max(MAX_PER_CLUB, owned_per_club.get(c, 0)) if allow_club_overage
               else MAX_PER_CLUB)
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= cap
    for i in ids:
        prob += start[i] <= squad[i]
    # Opt-in only, and only the wildcard rebuild opts in: a player below the
    # floor may be owned, but then he has to start. Weekly plans leave this at
    # 0 -- one free transfer cannot repair a bench, so a floor there would just
    # make the solve infeasible or force a worse move.
    if bench_floor > 0 and BENCH_FLOOR_COL in xp_df.columns:
        week = dict(zip(ids, xp_df[BENCH_FLOOR_COL].astype(float)))
        for i in ids:
            if week[i] < bench_floor:
                prob += squad[i] - start[i] <= 0
    # keep at least 15 - max_changes of the current squad
    prob += pulp.lpSum(squad[i] for i in ids if i in current_set) >= 15 - max_changes

    # No-good cuts, so the caller can ask for several DIFFERENT plans rather
    # than the same one repeatedly. See optimize.squad.enumerate_squads for why
    # near-identical candidates make a rank comparison meaningless.
    for combo in (excluded or []):
        present = [i for i in combo if i in squad]
        if present:
            drop = min(int(min_different), len(present))
            prob += pulp.lpSum(squad[i] for i in present) <= len(present) - drop

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
        return None
    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    # Untilted, for the same reason as optimize.squad: the plan is weighed
    # against a hit cost denominated in real points.
    gross = sum(raw_xp[i] for i in starters) + captain_bonus(
        cap_vars, captain_values(xp_df, ids, cfg, xp_col))
    return chosen, starters, gross


def _plan(current_set, solved, free_transfers, cfg) -> TransferPlan:
    chosen, starters, gross = solved
    actual = len(current_set - set(chosen))
    hit = max(0, actual - int(free_transfers)) * int(cfg.hit_cost)
    return TransferPlan(
        out_ids=sorted(current_set - set(chosen)),
        in_ids=sorted(set(chosen) - current_set),
        n_transfers=actual,
        hit_cost=hit,
        squad_ids=chosen,
        starting_ids=starters,
        gross_xp=round(gross, 3),
        net_xp=round(gross - hit, 3),
    )


def _budget_and_cost(xp_df, current_set, bank, selling_prices):
    price = dict(zip(xp_df["player_id"].astype(int), xp_df["price"].astype(float)))
    cost = dict(price)
    for pid in current_set:
        cost[pid] = float((selling_prices or {}).get(pid, price[pid]))
    return float(bank) + sum(cost[i] for i in current_set), cost


def _attribute_gain(plans: list["TransferPlan"]) -> None:
    """Record what each plan buys over simply holding, in place.

    The report quotes `gain`, so a plan that never has it filled in advertises
    itself as worth exactly nothing. The baseline is the 0-transfer plan, found
    by its transfer count rather than its position in the list: both callers
    happen to enumerate n=0 first, but a plan list that did not would otherwise
    silently measure every gain against the wrong squad.
    """
    hold = next((p for p in plans if p.n_transfers == 0), None)
    baseline = hold.net_xp if hold is not None else (plans[0].net_xp if plans else 0.0)
    for p in plans:
        p.baseline_xp = baseline
        p.gain = round(p.net_xp - baseline, 3)


def enumerate_transfer_plans(xp_df: pd.DataFrame, current_squad_ids: list[int],
                             bank: float, free_transfers: int, cfg,
                             xp_col: str = "xp_next5",
                             selling_prices: dict[int, float] | None = None,
                             k: int = 8, min_different: int = 1) -> list[TransferPlan]:
    """Several plausible transfer plans, for the rank layer to choose between.

    The weekly run had no equivalent of `optimize.squad.enumerate_squads`, so
    the distributional layer was never applied to it -- the plain
    expected-points objective it exists to replace was still deciding every
    gameweek on its own, in the one mode anybody runs weekly.

    Plans are enumerated in two passes. The first reserves one slot for the best
    plan at EVERY transfer count from 0 to free-plus-paid, so the rank layer
    always sees the whole decision -- hold, free moves, and each paid hit. Only
    then does the second pass spend the remaining quota on alternatives, deepest
    count first. `min_different` defaults to 1 rather than the 4 used for a
    squad rebuild -- a transfer plan IS a small change, and forcing candidates
    four players apart would only return plans nobody would consider.

    `k` is therefore a floor on breadth, not a hard ceiling: a `k` smaller than
    the number of transfer counts widens the decision rather than truncating it.
    """
    current_set = {int(i) for i in current_squad_ids}
    budget, cost = _budget_and_cost(xp_df, current_set, bank, selling_prices)
    max_n = int(free_transfers) + int(cfg.max_paid_hits)
    plans: list[TransferPlan] = []
    excluded: list[list[int]] = []

    def add(plan) -> bool:
        """Keep a plan unless an identical fifteen is already in the list."""
        if any(set(plan.squad_ids) == set(p.squad_ids) for p in plans):
            return False
        plans.append(plan)
        excluded.append(list(plan.squad_ids))
        return True

    # PASS 1 -- one reserved slot per transfer count.
    #
    # A single global quota filled in ascending order never got this far: the
    # hold plan took a slot, a large pool then supplied every remaining slot
    # with one-transfer alternatives, and the outer loop exited before n=2. With
    # one free transfer the rank layer was therefore never shown a paid hit, and
    # with banked transfers it was never shown a coordinated two-move
    # restructure. Reserving a slot per count costs at most max_n solves and
    # guarantees the whole decision is on the table.
    for n in range(0, max_n + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                        allow_club_overage=(n == 0))
        if solved is None:
            continue
        add(_plan(current_set, solved, free_transfers, cfg))

    # PASS 2 -- spend whatever quota is left on genuinely different alternatives,
    # deepest count first so the extra candidates are the ones the first pass
    # could not express.
    for n in range(max_n, -1, -1):
        while len(plans) < int(k):
            solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                            excluded=excluded, min_different=int(min_different),
                            allow_club_overage=(n == 0))
            if solved is None:
                break
            if not add(_plan(current_set, solved, free_transfers, cfg)):
                break
        if len(plans) >= int(k):
            break

    _attribute_gain(plans)
    return plans


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
    budget, cost = _budget_and_cost(xp_df, current_set, bank, selling_prices)

    options: list[TransferPlan] = []
    for n in range(0, int(free_transfers) + int(cfg.max_paid_hits) + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col, cost=cost,
                        allow_club_overage=(n == 0))
        if solved is None:
            continue
        options.append(_plan(current_set, solved, free_transfers, cfg))

    # The 0-transfer baseline is always solved and kept visible, so holding is
    # a candidate the others have to beat rather than an unpriced default.
    _attribute_gain(options)
    best = max(options, key=lambda o: o.net_xp)
    return best, options
