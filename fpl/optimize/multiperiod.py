"""Rolling multi-gameweek transfer optimisation.

Unlike :mod:`fpl.optimize.transfers`, this solver can wait to make a move and
can spend or roll free transfers later in the projection horizon. Prices are
held fixed because the model does not forecast price changes; the initial
squad's recorded selling values are nevertheless honoured until each player
is first sold.

The live pipeline keeps this behind an explicit configuration switch until its
terminal free-transfer value has been validated by sequential replay. The
replay exposes it as a challenger policy in the meantime.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd
import pulp

from ..config import FT_CAP
from .objective import event_columns, tilted_frame
from .squad import MAX_PER_CLUB, SQUAD_SPLIT, XI_MAX, XI_MIN, XI_SIZE
from .transfers import TransferPlan


@dataclass
class MultiPeriodWeek:
    event: int
    squad_ids: list[int]
    starting_ids: list[int]
    captain: int | None
    out_ids: list[int]
    in_ids: list[int]
    free_transfers_before: int
    free_transfers_after: int
    paid_transfers: int
    hit_cost: int
    bank_after: float
    gross_xp: float


@dataclass
class MultiPeriodPlan:
    weeks: list[MultiPeriodWeek] = field(default_factory=list)
    objective_xp: float = 0.0
    baseline_objective_xp: float = 0.0
    gain: float = 0.0
    terminal_free_transfers: int = 0
    terminal_ft_value: float = 0.0

    def first_week_plan(self) -> TransferPlan:
        """Present the executable first step through the existing report API."""
        if not self.weeks:
            raise ValueError("a multi-period plan must contain at least one gameweek")
        first = self.weeks[0]
        return TransferPlan(
            out_ids=list(first.out_ids),
            in_ids=list(first.in_ids),
            n_transfers=len(first.out_ids),
            hit_cost=int(first.hit_cost),
            squad_ids=list(first.squad_ids),
            starting_ids=list(first.starting_ids),
            gross_xp=round(self.objective_xp + first.hit_cost, 3),
            net_xp=round(self.objective_xp, 3),
            baseline_xp=round(self.baseline_objective_xp, 3),
            gain=round(self.gain, 3),
            strategy="multi-period",
            future_plan=[
                {
                    "event": w.event,
                    "out_ids": list(w.out_ids),
                    "in_ids": list(w.in_ids),
                    "bank_after": w.bank_after,
                    "free_transfers_after": w.free_transfers_after,
                    "hit_cost": w.hit_cost,
                }
                for w in self.weeks[1:]
            ],
            terminal_free_transfers=int(self.terminal_free_transfers),
            terminal_value=round(
                self.terminal_free_transfers * self.terminal_ft_value, 3),
        )


def _pruned_pool(xp_df: pd.DataFrame, current: set[int], cfg) -> pd.DataFrame:
    """Keep the best projected candidates plus every player already owned."""
    size = int(getattr(cfg, "multi_period_pool_size", 150))
    if size <= 0 or len(xp_df) <= size:
        return xp_df.copy().reset_index(drop=True)

    columns = event_columns(xp_df)
    score_col = "xp_horizon" if "xp_horizon" in xp_df.columns else columns[0][1]
    ranked = xp_df.sort_values(score_col, ascending=False)
    keep = set(ranked.head(size)["player_id"].astype(int)) | current
    # Global top-N can become position-thin in synthetic or early-season data.
    # A legal squad needs these minima even after all owned players are sold.
    for position, need in SQUAD_SPLIT.items():
        positional = ranked[ranked["position"] == position].head(max(need * 4, need))
        keep.update(positional["player_id"].astype(int))
    return xp_df[xp_df["player_id"].astype(int).isin(keep)].copy().reset_index(drop=True)


def _solve_path(
    xp_df: pd.DataFrame,
    current_squad_ids: list[int],
    bank: float,
    free_transfers: int,
    cfg,
    selling_prices: dict[int, float] | None,
    *,
    force_first_hold: bool = False,
) -> MultiPeriodPlan | None:
    columns = event_columns(xp_df)
    if not columns:
        raise ValueError("multi-period optimisation requires xp_gw{event} columns")

    current = {int(i) for i in current_squad_ids}
    if len(current) != sum(SQUAD_SPLIT.values()):
        raise ValueError("multi-period optimisation requires a 15-player current squad")
    missing = current - set(xp_df["player_id"].astype(int))
    if missing:
        raise ValueError(f"current squad players are missing from the projection: {sorted(missing)}")
    if not 0 <= int(free_transfers) <= FT_CAP:
        raise ValueError(f"free_transfers must be 0..{FT_CAP}, got {free_transfers}")

    pool = _pruned_pool(xp_df, current, cfg)
    ids = [int(i) for i in pool["player_id"]]
    events = [event for event, _ in columns]
    col_for = dict(columns)
    price = dict(zip(ids, pool["price"].astype(float)))
    position = dict(zip(ids, pool["position"].astype(str)))
    club = dict(zip(ids, pool["team"]))
    initial_sale = {
        i: float((selling_prices or {}).get(i, price[i])) for i in current
    }
    objective_col = "xp_horizon" if "xp_horizon" in pool.columns else columns[0][1]
    tilted = tilted_frame(pool, cfg, objective_col)
    raw = pool.set_index("player_id")
    tilted_by_id = tilted.set_index("player_id")
    decay = float(getattr(cfg, "horizon_decay", 1.0))
    hit_cost = int(getattr(cfg, "hit_cost", 4))
    max_paid = int(getattr(cfg, "max_paid_hits", 0))
    terminal_ft_value = float(getattr(cfg, "multi_period_ft_value", 1.5))
    max_moves = min(sum(SQUAD_SPLIT.values()), FT_CAP + max_paid)

    prob = pulp.LpProblem("fpl_multi_period", pulp.LpMaximize)
    owned = pulp.LpVariable.dicts("owned", (events, ids), cat="Binary")
    start = pulp.LpVariable.dicts("start", (events, ids), cat="Binary")
    tin = pulp.LpVariable.dicts("tin", (events, ids), cat="Binary")
    tout = pulp.LpVariable.dicts("tout", (events, ids), cat="Binary")
    bank_after = pulp.LpVariable.dicts("bank", events, lowBound=0)

    # Whether an initially-owned player is still held on the original purchase
    # basis. Once sold this stays zero, so a later repurchase sells at market.
    original = pulp.LpVariable.dicts("original", (events, sorted(current)), cat="Binary")
    original_sale = pulp.LpVariable.dicts(
        "original_sale", (events, sorted(current)), cat="Binary")

    # Exact FT transition table. A one-hot (balance-before, moves) state avoids
    # min/max relaxations that could invent free transfers or suppress hits.
    states: dict[int, dict[tuple[int, int], pulp.LpVariable]] = {}
    ft_before_expr = {}
    ft_after_expr = {}
    hits_expr = {}
    moves_expr = {}
    for offset, event in enumerate(events):
        valid = [
            (f, moves)
            for f in range(FT_CAP + 1)
            for moves in range(max_moves + 1)
            if moves <= f + max_paid
        ]
        states[event] = {
            state: pulp.LpVariable(f"ft_state_{event}_{state[0]}_{state[1]}", cat="Binary")
            for state in valid
        }
        prob += pulp.lpSum(states[event].values()) == 1
        ft_before_expr[event] = pulp.lpSum(
            f * var for (f, _), var in states[event].items())
        moves_expr[event] = pulp.lpSum(
            moves * var for (_, moves), var in states[event].items())
        hits_expr[event] = pulp.lpSum(
            max(0, moves - f) * var
            for (f, moves), var in states[event].items()
        )
        ft_after_expr[event] = pulp.lpSum(
            min(FT_CAP, max(0, f - moves) + 1) * var
            for (f, moves), var in states[event].items()
        )
        if offset == 0:
            prob += ft_before_expr[event] == int(free_transfers)
            if force_first_hold:
                prob += moves_expr[event] == 0
        else:
            prob += ft_before_expr[event] == ft_after_expr[events[offset - 1]]

    objective_terms = []
    captain_vars = {}
    bench_vars = {}
    for offset, event in enumerate(events):
        prev_event = events[offset - 1] if offset else None
        prob += pulp.lpSum(owned[event][i] for i in ids) == sum(SQUAD_SPLIT.values())
        prob += pulp.lpSum(start[event][i] for i in ids) == XI_SIZE
        prob += pulp.lpSum(tin[event][i] for i in ids) == moves_expr[event]
        prob += pulp.lpSum(tout[event][i] for i in ids) == moves_expr[event]

        for i in ids:
            before = owned[prev_event][i] if prev_event is not None else int(i in current)
            prob += owned[event][i] == before + tin[event][i] - tout[event][i]
            prob += tin[event][i] + tout[event][i] <= 1
            prob += start[event][i] <= owned[event][i]

        for pos, count in SQUAD_SPLIT.items():
            members = [i for i in ids if position[i] == pos]
            prob += pulp.lpSum(owned[event][i] for i in members) == count
            prob += pulp.lpSum(start[event][i] for i in members) >= XI_MIN[pos]
            prob += pulp.lpSum(start[event][i] for i in members) <= XI_MAX[pos]

        any_move = pulp.LpVariable(f"any_move_{event}", cat="Binary")
        prob += moves_expr[event] <= max_moves * any_move
        prob += moves_expr[event] >= any_move
        for team in set(club.values()):
            # Holding an inherited four-from-one-club squad is legal; making
            # any transfer requires the resulting squad to return to the cap.
            prob += pulp.lpSum(
                owned[event][i] for i in ids if club[i] == team
            ) <= MAX_PER_CLUB + 12 * (1 - any_move)

        sale_terms = [price[i] * tout[event][i] for i in ids]
        for i in sorted(current):
            if offset == 0:
                prob += original_sale[event][i] == tout[event][i]
                prob += original[event][i] == 1 - original_sale[event][i]
            else:
                prev_original = original[prev_event][i]
                prob += original_sale[event][i] <= tout[event][i]
                prob += original_sale[event][i] <= prev_original
                prob += original_sale[event][i] >= tout[event][i] + prev_original - 1
                prob += original[event][i] == prev_original - original_sale[event][i]
            sale_terms.append(
                (initial_sale[i] - price[i]) * original_sale[event][i]
            )
        previous_bank = float(bank) if offset == 0 else bank_after[prev_event]
        prob += bank_after[event] == (
            previous_bank + pulp.lpSum(sale_terms)
            - pulp.lpSum(price[i] * tin[event][i] for i in ids)
        )

        values = {i: float(tilted_by_id.loc[i, col_for[event]]) for i in ids}
        # Use the whole pruned pool. A top-60 shortcut can make the model
        # infeasible when an inherited low-projection squad contains none of
        # those candidates and cannot legally replace enough players at once.
        captain_pool = ids
        captain_vars[event] = pulp.LpVariable.dicts(
            f"captain_{event}", captain_pool, lowBound=0, upBound=1)
        prob += pulp.lpSum(captain_vars[event].values()) == 1
        for i in captain_pool:
            prob += captain_vars[event][i] <= start[event][i]

        outfield = [i for i in ids if position[i] != "GKP"]
        keepers = [i for i in ids if position[i] == "GKP"]
        bench_vars[event] = {
            slot: pulp.LpVariable.dicts(
                f"bench_{event}_{slot}", outfield, lowBound=0, upBound=1)
            for slot in range(3)
        }
        for slot in range(3):
            prob += pulp.lpSum(bench_vars[event][slot].values()) == 1
        for i in outfield:
            prob += pulp.lpSum(
                bench_vars[event][slot][i] for slot in range(3)
            ) == owned[event][i] - start[event][i]

        weights = [float(w) for w in cfg.bench_weight]
        week_points = (
            pulp.lpSum(values[i] * start[event][i] for i in ids)
            + pulp.lpSum(values[i] * captain_vars[event][i] for i in captain_pool)
            + pulp.lpSum(
                weights[slot] * values[i] * bench_vars[event][slot][i]
                for slot in range(3) for i in outfield
            )
            + pulp.lpSum(
                weights[3] * values[i] * (owned[event][i] - start[event][i])
                for i in keepers
            )
            - hit_cost * hits_expr[event]
            # Resolve exact score/FT ties in favour of doing less. Without a
            # tiny tie-breaker, CBC may churn equal-price/equal-xP players even
            # though the resulting recommendation buys nothing.
            - 0.0001 * moves_expr[event]
        )
        objective_terms.append((decay ** offset) * week_points)

    terminal_ft = ft_after_expr[events[-1]]
    prob += pulp.lpSum(objective_terms) + terminal_ft_value * terminal_ft
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    weeks = []
    for event in events:
        chosen = [i for i in ids if (owned[event][i].value() or 0) > 0.5]
        starters = [i for i in ids if (start[event][i].value() or 0) > 0.5]
        ins = [i for i in ids if (tin[event][i].value() or 0) > 0.5]
        outs = [i for i in ids if (tout[event][i].value() or 0) > 0.5]
        captain = next(
            (i for i, var in captain_vars[event].items() if (var.value() or 0) > 0.5),
            None,
        )
        state = next(
            (pair for pair, var in states[event].items() if (var.value() or 0) > 0.5),
            None,
        )
        if state is None:
            raise RuntimeError(f"solver returned no FT state for GW{event}")
        ft_before, moves = state
        paid = max(0, moves - ft_before)
        ft_after = min(FT_CAP, max(0, ft_before - moves) + 1)

        raw_values = {i: float(raw.loc[i, col_for[event]]) for i in ids}
        gross = sum(raw_values[i] for i in starters)
        if captain is not None:
            gross += raw_values[captain]
        for slot in range(3):
            for i, var in bench_vars[event][slot].items():
                if (var.value() or 0) > 0.5:
                    gross += float(cfg.bench_weight[slot]) * raw_values[i]
        gross += sum(
            float(cfg.bench_weight[3]) * raw_values[i]
            for i in ids
            if position[i] == "GKP" and i in chosen and i not in starters
        )
        weeks.append(MultiPeriodWeek(
            event=int(event), squad_ids=chosen, starting_ids=starters,
            captain=None if captain is None else int(captain),
            out_ids=sorted(outs), in_ids=sorted(ins),
            free_transfers_before=int(ft_before), free_transfers_after=int(ft_after),
            paid_transfers=int(paid), hit_cost=int(paid * hit_cost),
            bank_after=round(float(bank_after[event].value()), 1),
            gross_xp=round(gross, 3),
        ))

    # The MILP may optimise a bounded ownership tilt, but reported xP must stay
    # in forecast units. Reconstruct it from the untilted weekly totals, just as
    # the one-period solvers do.
    objective = sum(
        decay ** offset * (week.gross_xp - week.hit_cost)
        for offset, week in enumerate(weeks)
    ) + terminal_ft_value * weeks[-1].free_transfers_after
    return MultiPeriodPlan(
        weeks=weeks,
        objective_xp=round(objective, 3),
        terminal_free_transfers=weeks[-1].free_transfers_after,
        terminal_ft_value=terminal_ft_value,
    )


def optimize_multi_period(
    xp_df: pd.DataFrame,
    current_squad_ids: list[int],
    bank: float,
    free_transfers: int,
    cfg,
    selling_prices: dict[int, float] | None = None,
) -> MultiPeriodPlan:
    """Return the optimal rolling path and its value over waiting one week.

    The baseline is not "never transfer". It forces only the first gameweek to
    be a hold and then re-optimises the remaining path, which is the relevant
    opportunity-cost comparison for deciding whether to spend an FT now.

    The MILP itself maximises ownership-tilted projections (see
    `optimize.objective.tilted_frame`) so a `differential`/`template` profile
    can break ties among close options, but `objective_xp` here is
    reconstructed in raw forecast units -- the same split the one-period
    solver keeps in `optimize.transfers._solve`. There, the wait/hold plan is
    just another candidate in the pool `_choose_transfers` maximises on raw
    `net_xp`, so a tilt-driven pick that turns out raw-negative can never
    win. The rolling solver has no such shared pool, so that guard has to be
    explicit here: a `chosen` path whose raw gain against holding is negative
    must not be reported as the recommendation, no matter how much the tilt
    favoured it internally.

    The comparator this is judged against must itself be raw-optimal, not
    just tilt-optimal-and-forced-to-hold-week-one: solving `wait` with `cfg`
    unchanged reuses the same tilt for every week after the forced hold, so
    its "raw" total is only the best a TILTED path can reach, not the best a
    raw one can. With tilt off, `tilted_frame` is the identity (see its
    early-exit when every factor is 1.0), so a solve with `ownership_weight`
    zeroed maximises exactly the quantity reported as raw xp -- that is the
    true ceiling `chosen` must clear, and always at least matches (never
    trails) the tilted wait's raw total.
    """
    chosen = _solve_path(
        xp_df, current_squad_ids, bank, free_transfers, cfg, selling_prices)
    if chosen is None:
        raise ValueError("multi-period transfer optimisation is infeasible")
    raw_cfg = replace(cfg, ownership_weight=0.0)
    wait = _solve_path(
        xp_df, current_squad_ids, bank, free_transfers, raw_cfg, selling_prices,
        force_first_hold=True)
    if wait is None:
        raise ValueError("multi-period wait-one-week baseline is infeasible")
    chosen.baseline_objective_xp = wait.objective_xp
    chosen.gain = round(chosen.objective_xp - wait.objective_xp, 3)
    if chosen.gain < 0:
        wait.baseline_objective_xp = wait.objective_xp
        wait.gain = 0.0
        return wait
    return chosen
