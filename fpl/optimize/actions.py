"""What each chip actually DOES to the squad.

The advisor used to be bolted on after optimization: transfers and lineup were
finalised first, and a Free Hit recommendation then came back attached to the
ordinary permanent transfer plan -- the one thing a Free Hit is not. A Wildcard
was worse: it solved a full rebuild purely to extract a scalar surplus, threw
the rebuilt squad away, and returned the limited-transfer squad whose objective
still included hit costs. Confirming either applied a team the named chip had
never chosen.

So each chip is modelled here as a COMPLETE action -- squad, XI, transfers and
the points it is worth. `optimize.chips` still decides the timing and
`fpl.chips` the legality; this module decides what happens when one fires.
"""
from dataclasses import dataclass, field

from .transfers import TransferPlan, _budget_and_cost, _plan, _solve

SQUAD_SIZE = 15
# A Free Hit squad exists for exactly one gameweek, so it is chosen on the
# single-week projection. Choosing it on the discounted horizon buys players
# for weeks the squad will have ceased to exist in.
ONE_WEEK_COL = "xp_next1"


@dataclass
class ChipAction:
    """One complete, legal thing the manager could do this gameweek."""
    chip: str | None
    squad_ids: list[int] = field(default_factory=list)
    starting_ids: list[int] = field(default_factory=list)
    transfers: TransferPlan | None = None
    # Points this action is worth, in whatever currency the caller compares on.
    value: float = 0.0
    # True when FPL takes the squad back at the next deadline (Free Hit only),
    # which is what decides whether local state may overwrite the permanent 15.
    temporary: bool = False
    detail: str = ""


def _rebuild(xp_df, cfg, current_squad, bank, selling, xp_col, bench_floor):
    """Solve for a full 15 with every player in play, on the money actually held.

    Unlimited transfers is exactly the weekly solve with `max_changes` at 15 --
    same objective, same budget, same selling prices -- which is why this goes
    through `transfers._solve` rather than `optimize_squad`. Routing it through
    the squad builder would compare against a fresh `cfg.budget` the manager
    does not have.
    """
    current = {int(i) for i in current_squad}
    budget, cost = _budget_and_cost(xp_df, current, bank, selling)
    solved = _solve(xp_df, current, budget, SQUAD_SIZE, cfg, xp_col, cost=cost,
                    bench_floor=bench_floor)
    if solved is None and bench_floor > 0:
        # A floor no budget can satisfy must not take the chip down with it.
        solved = _solve(xp_df, current, budget, SQUAD_SIZE, cfg, xp_col, cost=cost)
    return solved, current


def wildcard_action(xp_df, cfg, current_squad, bank: float, selling: dict,
                    xp_col: str = "xp_horizon") -> ChipAction | None:
    """The permanent rebuild a Wildcard buys, and what it is worth.

    Valued on the HORIZON, not one week: a Wildcard squad is kept, so the gain
    it buys is collected over every gameweek the projection reaches.

    `n_transfers` is reported for display only. The chip makes them free, so
    the plan is built with `free_transfers` at 15 and carries no hit.
    """
    bench_floor = float(getattr(cfg, "bench_floor_xp", 0.0) or 0.0)
    solved, current = _rebuild(xp_df, cfg, current_squad, bank, selling, xp_col,
                               bench_floor)
    if solved is None:
        return None
    plan = _plan(current, solved, SQUAD_SIZE, cfg)
    return ChipAction(
        chip="wildcard",
        squad_ids=list(plan.squad_ids),
        starting_ids=list(plan.starting_ids),
        transfers=plan,
        value=round(float(plan.net_xp), 3),
        temporary=False,
        detail=f"unlimited free transfers, changing {plan.n_transfers} of your 15",
    )


def freehit_action(xp_df, cfg, current_squad, bank: float,
                   selling: dict) -> ChipAction | None:
    """The one-week squad a Free Hit fields, and what it scores that week.

    Built on the same money as a Wildcard -- the chip does not grant a fresh
    100m -- but optimised on `xp_next1`, and flagged `temporary` so the caller
    knows FPL will take all fifteen back at the next deadline.

    No bench floor: a Free Hit bench never plays, so spending XI budget to fill
    it would be spending it for nothing.
    """
    solved, current = _rebuild(xp_df, cfg, current_squad, bank, selling,
                               ONE_WEEK_COL, bench_floor=0.0)
    if solved is None:
        return None
    plan = _plan(current, solved, SQUAD_SIZE, cfg)
    frame = xp_df.set_index("player_id")
    value = float(frame.loc[list(plan.starting_ids), ONE_WEEK_COL].sum())
    return ChipAction(
        chip="freehit",
        squad_ids=list(plan.squad_ids),
        starting_ids=list(plan.starting_ids),
        transfers=plan,
        value=round(value, 3),
        temporary=True,
        detail=f"a one-week squad changing {plan.n_transfers} of your 15",
    )


def bench_boost_value(lineup, xp_df, xp_col: str = ONE_WEEK_COL) -> float:
    """What a Bench Boost adds: the four players who are ACTUALLY benched.

    The advisor approximated this as the four numerically lowest projections in
    the squad, which need not be a legal bench under formation constraints --
    a 3-5-2 benches a defender the approximation would have started.
    """
    frame = xp_df.set_index("player_id")
    bench = [int(i) for i in lineup.bench]
    if not bench:
        return 0.0
    return round(float(frame.loc[bench, xp_col].sum()), 3)


def triple_captain_value(lineup, xp_df, xp_col: str = ONE_WEEK_COL) -> float:
    """What a Triple Captain adds: ONE more armband return.

    The armband already pays double; the chip pays a third multiple. `xp_col`
    is the UNCONDITIONAL expectation -- it already contains the chance the
    captain does not appear -- so the extra multiple is worth exactly that
    figure when he plays. Multiplying it by `p_play` again, which an earlier
    version did, discounted him twice.

    And the chip is not wasted when he does not play: FPL's rules pass the
    triple to the vice-captain, so in those scenarios the third multiple lands
    on the vice instead. Under the usual independence approximation that is
    `(1 - p_play) * vice_xp`, the same shape `choose_captain` already uses for
    ordinary vice inheritance.
    """
    frame = xp_df.set_index("player_id")
    captain = int(lineup.captain)
    captain_xp = float(frame.loc[captain, xp_col])
    p_play = (float(frame.loc[captain, "p_play"])
              if "p_play" in frame.columns else 1.0)
    vice_xp = (float(frame.loc[int(lineup.vice), xp_col])
               if lineup.vice is not None else 0.0)
    return round(captain_xp + (1.0 - p_play) * vice_xp, 3)
