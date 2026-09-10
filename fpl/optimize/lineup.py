"""Starting XI presentation: formation, bench order, captain and vice."""
from dataclasses import dataclass
import pandas as pd

from .squad import Squad


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


def build_lineup(squad: Squad, xp_df: pd.DataFrame, xp_col: str = "xp_next1") -> Lineup:
    df = xp_df.set_index("player_id")
    xi = list(squad.starting_ids)
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
