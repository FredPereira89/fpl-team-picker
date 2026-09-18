"""Replay one manager's season, decision by decision.

The existing walk-forward harness calls `optimize_squad` from scratch every
gameweek. That is a free Wildcard every week, on unlimited budget, with no
transfer history, no hits, no free-transfer balance and no chips -- and its
result was being compared against an FPL field average that contains all of
those things. It is an oracle upper bound, not a backtest, and a policy that
looks good under it may be impossible to execute.

This module replays STATE instead. One squad, one bank, one set of purchase
prices, one free-transfer balance and one chip inventory, carried gameweek to
gameweek exactly as the live tool carries them -- the same `ft_after_moves`,
the same `selling_price`, the same `chip_available`, the same
`realised_score`. A bug in the live rules is therefore a bug here too, which is
the point: a replay that reimplements the rules proves nothing about the tool.

The free rebuild is kept, as `oracle_rebuild_policy`, and labelled a ceiling
wherever it is reported.
"""
from dataclasses import dataclass, field, replace as _replace

import pandas as pd

from ..chips import chip_available
from ..optimize.lineup import build_lineup
from ..optimize.objective import event_columns
from ..optimize.squad import Squad, optimize_squad
from ..optimize.transfers import (optimize_transfers, selling_price, bank_after,
                                  TransferPlan)
from ..state import State, ft_after_moves, FT_CAP
from .walkforward import realised_score

SQUAD_SIZE = 15
# Chips that make a gameweek's transfers free, and whose squad therefore costs
# no hit however many players changed.
FREE_TRANSFER_CHIPS = ("wildcard", "freehit")


@dataclass
class ManagerState:
    """Everything a real FPL manager carries from one gameweek to the next."""
    squad: list[int]
    bank: float = 0.0
    purchase_prices: dict[int, float] = field(default_factory=dict)
    free_transfers: int = 1
    chip_events: list[dict] = field(default_factory=list)
    # The permanent squad a Free Hit is temporarily replacing. Mirrors
    # fpl.state.State: without it a replayed Free Hit keeps its one-week team.
    base_squad: list[int] = field(default_factory=list)
    base_bank: float = 0.0
    base_purchase_prices: dict[int, float] = field(default_factory=dict)
    freehit_event: int | None = None
    points: float = 0.0
    hits: int = 0

    def copy(self) -> "ManagerState":
        return _replace(
            self,
            squad=list(self.squad),
            purchase_prices=dict(self.purchase_prices),
            chip_events=[dict(c) for c in self.chip_events],
            base_squad=list(self.base_squad),
            base_purchase_prices=dict(self.base_purchase_prices),
        )


@dataclass
class Decision:
    """What a policy chose for one gameweek.

    `free_transfers` overrides the state's balance for this gameweek only. It
    exists for the oracle, whose whole premise is that the rules do not apply:
    charging its weekly rebuild 108 points of hits made the "ceiling" score
    below doing nothing, which is not a ceiling, it is a different mistake.
    Every executable policy leaves this None and is charged normally.
    """
    squad_ids: list[int]
    starting_ids: list[int]
    chip: str | None = None
    free_transfers: int | None = None
    # The armband as the policy actually set it. Left None, scoring falls back
    # to the two highest projections -- which is not the production decision.
    captain: int | None = None
    vice: int | None = None


@dataclass
class GameweekResult:
    gw: int
    points: float            # net of hits
    gross_points: float
    hit_cost: int
    transfers: int
    chip: str | None
    squad: list[int]
    xi: list[int]
    captain: int
    free_transfers_after: int
    bank_after: float


def _prices(xp: pd.DataFrame) -> dict[int, float]:
    return dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))


def selling_values(state: ManagerState, xp: pd.DataFrame) -> dict[int, float]:
    """What FPL would pay for each owned player, at recorded purchase prices."""
    price = _prices(xp)
    return {int(i): selling_price(state.purchase_prices.get(int(i), price[int(i)]),
                                  price[int(i)])
            for i in state.squad if int(i) in price}


# --- policies ---------------------------------------------------------------
#
# A policy sees the projection, the state and the gameweek, and returns one
# Decision. It may not mutate the state: the walk below owns that, so every
# policy is charged for its transfers by the same code.

def one_week_frame(xp: pd.DataFrame) -> pd.DataFrame:
    """The projection with only the CURRENT event's per-gameweek column.

    The shared objective values the armband in every xp_gw column it can see.
    A policy that is deciding one week -- the hold XI, the oracle rebuild --
    must not be nudged by who captains well next month.
    """
    cols = event_columns(xp)
    if len(cols) <= 1:
        return xp
    return xp.drop(columns=[c for _, c in cols[1:]])


def _with_armband(decision: Decision, xp: pd.DataFrame) -> Decision:
    """Set captain and vice through the LIVE lineup function.

    `choose_captain` discounts a captain for his chance of not appearing, so
    the production armband can differ from the raw top projection. Every
    executable policy goes through here so the replay scores the decision the
    tool would actually recommend.
    """
    lineup = build_lineup(Squad(list(decision.squad_ids),
                                list(decision.starting_ids), 0.0, 0.0), xp)
    decision.captain, decision.vice = int(lineup.captain), int(lineup.vice)
    return decision


def hold_policy(xp, state, gw, cfg) -> Decision:
    """Make no transfer, ever. The baseline every other policy must beat."""
    squad = [int(i) for i in state.squad]
    week = one_week_frame(xp)
    best, _ = optimize_transfers(week, squad, state.bank, 0, cfg,
                                 xp_col="xp_next1",
                                 selling_prices=selling_values(state, xp))
    return _with_armband(Decision(list(best.squad_ids), list(best.starting_ids)), xp)


def expected_points_policy(xp, state, gw, cfg) -> Decision:
    """The production transfer policy: discounted horizon net points.

    `xp` must carry the configured multi-gameweek horizon -- `xp_horizon` and
    the `xp_gw*` columns as `build_xp` produces them for a live run. Handing
    this policy a one-week frame turns it into a greedy weekly chooser and the
    replay then measures something the live tool never does.
    """
    best, _ = optimize_transfers(xp, [int(i) for i in state.squad], state.bank,
                                 int(state.free_transfers), cfg,
                                 xp_col="xp_horizon",
                                 selling_prices=selling_values(state, xp))
    return _with_armband(Decision(list(best.squad_ids), list(best.starting_ids)), xp)


def oracle_rebuild_policy(xp, state, gw, cfg) -> Decision:
    """A free Wildcard every single gameweek, on a fresh budget.

    NOT EXECUTABLE. It ignores the squad, the bank, purchase prices, the
    free-transfer balance and the chip inventory. It is reported only as a
    ceiling -- the score a manager could reach if the rules did not apply --
    because the old harness reported exactly this as if it were a backtest.
    """
    squad = optimize_squad(one_week_frame(xp), cfg, xp_col="xp_next1")
    return _with_armband(Decision(list(squad.player_ids), list(squad.starting_ids),
                                  free_transfers=SQUAD_SIZE), xp)


def _apply_freehit_restoration(state: ManagerState, gw: int) -> ManagerState:
    """Undo a Free Hit played in an earlier gameweek.

    FPL restores the pre-chip squad, bank and purchase prices at the next
    deadline, and any money the chip week freed up is lost.
    """
    if state.freehit_event is None or int(gw) <= int(state.freehit_event):
        return state
    if not state.base_squad:
        return state
    out = state.copy()
    out.squad = list(state.base_squad)
    out.bank = float(state.base_bank)
    out.purchase_prices = dict(state.base_purchase_prices)
    out.base_squad, out.base_bank, out.base_purchase_prices = [], 0.0, {}
    out.freehit_event = None
    return out


def step(xp: pd.DataFrame, actuals: pd.DataFrame, state: ManagerState, gw: int,
         cfg, policy) -> tuple[GameweekResult, ManagerState]:
    """Play one gameweek: decide, charge, score, and advance the state."""
    state = _apply_freehit_restoration(state, gw)
    before = state.copy()

    decision = policy(xp, before, gw, cfg)
    chip = decision.chip
    if chip is not None and not chip_available(chip, gw, before.chip_events):
        # A policy may propose an illegal chip; the replay must not let it play
        # one, exactly as the live tool would not.
        chip = None

    squad = [int(i) for i in decision.squad_ids]
    transfers = len(set(before.squad) - set(squad))
    if chip in FREE_TRANSFER_CHIPS:
        free = SQUAD_SIZE
    elif decision.free_transfers is not None:
        free = int(decision.free_transfers)      # the oracle, and only the oracle
    else:
        free = int(before.free_transfers)
    hit_cost = max(0, transfers - free) * int(cfg.hit_cost)

    price = _prices(xp)
    frame = xp.set_index("player_id").join(
        actuals.set_index("player_id")[["actual", "minutes"]], how="left")
    frame[["actual", "minutes"]] = frame[["actual", "minutes"]].fillna(0.0)
    scored = realised_score(squad, list(decision.starting_ids), frame,
                            captain=decision.captain, vice=decision.vice)

    after = before.copy()
    # Bank and purchase prices move before the chip bookkeeping, because a Free
    # Hit's temporary squad still has to be priced to be legal.
    after.bank = bank_after(before.bank, before.squad, squad, price,
                            before.purchase_prices)
    after.purchase_prices = {int(p): float(before.purchase_prices.get(int(p),
                                                                     price[int(p)]))
                             for p in squad}
    after.squad = squad

    if chip == "freehit":
        after.base_squad = list(before.squad)
        after.base_bank = float(before.bank)
        after.base_purchase_prices = dict(before.purchase_prices)
        after.freehit_event = int(gw)

    if chip is not None:
        after.chip_events = before.chip_events + [{"chip": chip, "event": int(gw)}]

    # The SAME function the live tool uses. Reimplementing the rule here would
    # mean the replay could pass while production was wrong.
    _, next_ft = ft_after_moves(State(free_transfers=int(before.free_transfers)),
                                transfers_made=transfers, chip=chip)
    after.free_transfers = min(FT_CAP, int(next_ft))

    gross = float(scored["points"])
    if chip == "benchboost":
        bench = [p for p in squad if p not in set(scored["xi"])]
        gross += float(sum(frame.loc[p, "actual"] for p in bench if p in frame.index))
    elif chip == "triplecaptain":
        gross += float(frame.loc[scored["captain"], "actual"])

    net = gross - hit_cost
    after.points = before.points + net
    after.hits = before.hits + hit_cost

    return GameweekResult(
        gw=int(gw), points=net, gross_points=gross, hit_cost=int(hit_cost),
        transfers=int(transfers), chip=chip, squad=squad, xi=list(scored["xi"]),
        captain=int(scored["captain"]),
        free_transfers_after=int(after.free_transfers),
        bank_after=float(after.bank),
    ), after


def replay_season(xp_by_gw: dict[int, pd.DataFrame], actuals: pd.DataFrame,
                  state: ManagerState, cfg, policy,
                  gameweeks=None, cfg_by_gw: dict | None = None
                  ) -> tuple[list[GameweekResult], ManagerState]:
    """Walk a manager's state through every gameweek, in order.

    `cfg_by_gw` supplies the configuration each gameweek was actually decided
    under (from its snapshot); `cfg` covers the rest. Running every historical
    week under today's settings reconstructs decisions the tool never made.
    """
    results = []
    order = sorted(gameweeks if gameweeks is not None else xp_by_gw)
    for gw in order:
        xp = xp_by_gw.get(int(gw))
        if xp is None:
            continue
        gw_actuals = actuals[actuals["round"] == int(gw)] if "round" in actuals \
            else actuals
        week_cfg = (cfg_by_gw or {}).get(int(gw), cfg)
        result, state = step(xp, gw_actuals, state, int(gw), week_cfg, policy)
        results.append(result)
    return results, state


# Policies that describe something a manager could actually do. The oracle is
# excluded on purpose, and every report has to say so.
EXECUTABLE_POLICIES = {
    "hold": hold_policy,
    "expected": expected_points_policy,
}
ORACLE_POLICIES = {"oracle": oracle_rebuild_policy}


def compare_policies(xp_by_gw, actuals, initial: ManagerState, cfg,
                     policies=None, field_average=None,
                     gameweeks=None, cfg_by_gw: dict | None = None) -> pd.DataFrame:
    """One row per policy: what it scored, what it paid, and whether it is real.

    `executable` is the column that matters. The old harness reported a free
    weekly rebuild against the FPL average without it, which is a comparison
    between a manager and a manager who does not have to obey the rules.
    """
    chosen = dict(policies or {**EXECUTABLE_POLICIES, **ORACLE_POLICIES})
    rows = []
    for name, policy in chosen.items():
        results, final = replay_season(xp_by_gw, actuals, initial.copy(), cfg,
                                       policy, gameweeks=gameweeks,
                                       cfg_by_gw=cfg_by_gw)
        weeks = [r.gw for r in results]
        edge = None
        if field_average:
            deltas = [r.points - float(field_average.get(r.gw, 0.0)) for r in results]
            edge = sum(deltas) / len(deltas) if deltas else None
        rows.append({
            "policy": name,
            "executable": name not in ORACLE_POLICIES,
            "gameweeks": len(weeks),
            "points": round(final.points, 1),
            "hits_paid": int(final.hits),
            "transfers": int(sum(r.transfers for r in results)),
            "chips_used": len(final.chip_events),
            "mean_edge": None if edge is None else round(edge, 2),
        })
    return pd.DataFrame(rows).sort_values(
        ["executable", "points"], ascending=[False, False]).reset_index(drop=True)
