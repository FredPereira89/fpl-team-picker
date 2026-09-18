"""Starting XI presentation: formation, bench order, captain and vice."""
from dataclasses import dataclass
import pandas as pd

from itertools import product

from .squad import Squad, XI_MIN, XI_MAX, XI_SIZE


def best_xi(squad_ids: list[int], xp_df: pd.DataFrame,
            xp_col: str = "xp_next1") -> list[int]:
    """The best legal eleven from a fifteen, exactly, for ONE gameweek.

    The solver picks a single XI to serve the whole projection horizon, so the
    XI it hands back can bench a player who is clearly the better pick THIS
    week because someone else is worth more as a horizon-long starter. The
    report is for this week, and this is that solve: every legal formation is
    enumerated (one keeper, 3-5 defenders, 2-5 midfielders, 1-3 forwards,
    eleven in all) and within a formation the top-k by projection per position
    is optimal for a sum objective, so the enumeration is exact and cheap.
    """
    frame = xp_df.set_index("player_id")
    by_pos: dict[str, list[int]] = {}
    for pid in squad_ids:
        pid = int(pid)
        by_pos.setdefault(str(frame.loc[pid, "position"]), []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: (-float(frame.loc[p, xp_col]), p))

    best, best_value = None, float("-inf")
    for d, m, f in product(range(XI_MIN["DEF"], XI_MAX["DEF"] + 1),
                           range(XI_MIN["MID"], XI_MAX["MID"] + 1),
                           range(XI_MIN["FWD"], XI_MAX["FWD"] + 1)):
        if 1 + d + m + f != XI_SIZE:
            continue
        need = {"GKP": 1, "DEF": d, "MID": m, "FWD": f}
        if any(len(by_pos.get(pos, [])) < n for pos, n in need.items()):
            continue
        xi = [p for pos, n in need.items() for p in by_pos[pos][:n]]
        value = sum(float(frame.loc[p, xp_col]) for p in xi)
        if value > best_value:
            best, best_value = xi, value
    if best is None:
        raise ValueError("no legal eleven can be formed from this squad")
    return best


@dataclass
class Lineup:
    xi: list[int]
    bench: list[int]
    formation: str
    captain: int
    vice: int
    xp: float


def bench_order(bench_ids: list[int], position: dict[int, str],
                xp: dict[int, float]) -> list[int]:
    """Reserve keeper first -- he can only cover the keeper -- then by projection."""
    return sorted(bench_ids, key=lambda p: (position[p] != "GKP", -float(xp[p])))


def choose_captain(xi: list[int], xp: dict[int, float],
                   p_play: dict[int, float]) -> tuple[int, int]:
    """(captain, vice) maximising what the armband is actually worth.

    The armband pays the captain's points if he appears at all, and the VICE's
    points if he does not -- so its value is
        xp[captain] + P(captain does not appear) * xp[vice],
    which is why the vice matters and why the second term keys on the chance of
    NO APPEARANCE rather than of not starting: a captain who comes off the
    bench for ten minutes keeps the armband and the vice gets nothing.

    Picking the top two projections outright ignores that free re-roll. With
    every starter equally likely to play it gives the same answer, so this only
    departs from the old ordering where the risk is real.
    """
    candidates = [i for i in xi if p_play.get(i, 1.0) > 0] or list(xi)
    best = None
    for c in candidates:
        others = [i for i in xi if i != c]
        if not others:
            continue
        v = max(others, key=lambda i: (xp[i], -i))
        value = xp[c] + (1.0 - float(p_play.get(c, 1.0))) * xp[v]
        # Tie-break on the captain's own projection: with no rotation risk
        # anywhere, every pair scores the same and the best player takes it.
        key = (value, xp[c], -c)
        if best is None or key > best[0]:
            best = (key, c, v)
    return best[1], best[2]


def build_lineup(squad: Squad, xp_df: pd.DataFrame, xp_col: str = "xp_next1",
                 exact: bool = True) -> Lineup:
    """The week's lineup: XI, bench order, captain and vice.

    `exact` re-picks the eleven on `xp_col` for this week rather than reusing
    the solver's horizon XI -- see `best_xi`. Off only for callers that must
    present a specific eleven as given.
    """
    df = xp_df.set_index("player_id")
    xi = (best_xi(list(squad.player_ids), xp_df, xp_col) if exact
          else list(squad.starting_ids))
    bench_ids = [i for i in squad.player_ids if i not in set(xi)]

    bench = bench_order(bench_ids, df["position"].to_dict(), df[xp_col].to_dict())

    counts = df.loc[xi, "position"].value_counts()
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"

    xp = {i: float(df.loc[i, xp_col]) for i in xi}
    # A frame without p_play (anything built outside model.xp) reads as "every
    # starter appears", which collapses to the old top-two ordering.
    p_play = ({i: float(df.loc[i, "p_play"]) for i in xi}
              if "p_play" in df.columns else {})
    captain, vice = choose_captain(xi, xp, p_play)
    total = sum(xp.values()) + xp[captain]

    return Lineup(xi=xi, bench=bench, formation=formation,
                  captain=captain, vice=vice, xp=round(total, 3))
