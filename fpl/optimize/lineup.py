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


def build_lineup(squad: Squad, xp_df: pd.DataFrame, xp_col: str = "xp_next1") -> Lineup:
    df = xp_df.set_index("player_id")
    xi = list(squad.starting_ids)
    bench_ids = [i for i in squad.player_ids if i not in set(xi)]

    # The reserve keeper can only replace the keeper, so it always sits in slot 1.
    keepers = [i for i in bench_ids if df.loc[i, "position"] == "GKP"]
    outfield = [i for i in bench_ids if df.loc[i, "position"] != "GKP"]
    outfield.sort(key=lambda i: float(df.loc[i, xp_col]), reverse=True)
    bench = keepers + outfield

    counts = df.loc[xi, "position"].value_counts()
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"

    ranked = sorted(xi, key=lambda i: float(df.loc[i, xp_col]), reverse=True)
    captain, vice = ranked[0], ranked[1]
    total = sum(float(df.loc[i, xp_col]) for i in xi) + float(df.loc[captain, xp_col])

    return Lineup(xi=xi, bench=bench, formation=formation,
                  captain=captain, vice=vice, xp=round(total, 3))
