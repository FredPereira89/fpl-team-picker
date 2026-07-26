"""Chip advisor - heuristic flags, never an unexplained auto-recommendation."""
from dataclasses import dataclass
import pandas as pd

BENCH_BOOST_MIN_XP = 2.5
TRIPLE_CAPTAIN_MIN_XP = 9.0
FREE_HIT_MIN_BLANKS = 3
WILDCARD_MIN_PROBLEMS = 4
PROBLEM_MARKERS = ("Unavailable", "Doubtful")


@dataclass
class ChipAdvice:
    chip: str | None
    reason: str


def advise_chips(xp_df: pd.DataFrame, lineup, squad_ids: list[int],
                 counts: pd.DataFrame, team_by_player: dict[int, int],
                 from_event: int, chips_used: list[str]) -> ChipAdvice:
    df = xp_df.set_index("player_id")
    used = set(chips_used or [])
    n_by_team = {
        int(r["team_id"]): int(r["n_fixtures"])
        for _, r in counts[counts["event"] == from_event].iterrows()
    }

    def fixtures_for(pid: int) -> int:
        return n_by_team.get(team_by_player.get(int(pid), -1), 1)

    blanks = sum(1 for pid in squad_ids if fixtures_for(pid) == 0)
    problems = sum(
        1 for pid in squad_ids
        if any(m in f for f in df.loc[pid, "flags"] for m in PROBLEM_MARKERS)
    )
    cap_xp = float(df.loc[lineup.captain, "xp_next1"])
    cap_fixtures = fixtures_for(lineup.captain)
    bench_xp = [float(df.loc[pid, "xp_next1"]) for pid in lineup.bench]

    free_hit_ok = blanks >= FREE_HIT_MIN_BLANKS
    wildcard_ok = problems >= WILDCARD_MIN_PROBLEMS
    triple_ok = cap_fixtures >= 2 or cap_xp >= TRIPLE_CAPTAIN_MIN_XP
    bench_ok = bool(bench_xp) and min(bench_xp) >= BENCH_BOOST_MIN_XP

    if free_hit_ok and "freehit" not in used:
        return ChipAdvice("freehit", (
            f"{blanks} of your 15 have no fixture this gameweek. A Free Hit fields a "
            f"one-week replacement squad, but you lose it for a future blank or double "
            f"— only worth it if you can't cover the gap with transfers."
        ))

    if wildcard_ok and "wildcard" not in used:
        return ChipAdvice("wildcard", (
            f"{problems} players carry injury or rotation flags. A Wildcard fixes them "
            f"all at once with unlimited free transfers, but spends a chip you may want "
            f"later for a fixture swing."
        ))

    if triple_ok and "triplecaptain" not in used:
        detail = "a double gameweek" if cap_fixtures >= 2 else "an outstanding single fixture"
        return ChipAdvice("triplecaptain", (
            f"{df.loc[lineup.captain, 'web_name']} has {detail} (xP {cap_xp:.1f}). "
            f"Triple Captain turns that into 3x, but a blank or an early substitution "
            f"wastes the chip entirely."
        ))

    if bench_ok and "benchboost" not in used:
        return ChipAdvice("benchboost", (
            f"All four bench players project at {min(bench_xp):.1f}+ xP "
            f"({sum(bench_xp):.1f} total). Bench Boost banks that, but a stronger bench "
            f"week may come later — especially in a double gameweek."
        ))

    blocked = []
    if free_hit_ok and "freehit" in used:
        blocked.append(f"{blanks} players have a blank fixture — Free Hit-worthy, but already used")
    if wildcard_ok and "wildcard" in used:
        blocked.append(f"{problems} players carry injury/rotation flags — Wildcard-worthy, but already used")
    if triple_ok and "triplecaptain" in used:
        blocked.append("your captain has a standout week — Triple-Captain-worthy, but already used")
    if bench_ok and "benchboost" in used:
        blocked.append("your bench projects strongly — Bench-Boost-worthy, but already used")

    if blocked:
        return ChipAdvice(None, "No chip available this week — " + "; ".join(blocked) + ".")

    return ChipAdvice(None, (
        "No chip recommended this week — the bench is too weak for a Bench Boost, "
        "no standout double-gameweek captain, and the squad has no cluster of blanks "
        "or injuries needing a Wildcard or Free Hit."
    ))
