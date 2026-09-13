"""Chip advisor - heuristic flags, never an unexplained auto-recommendation."""
from dataclasses import dataclass
import pandas as pd

from .objective import event_columns

BENCH_BOOST_MIN_XP = 2.5
TRIPLE_CAPTAIN_MIN_XP = 9.0
FREE_HIT_MIN_BLANKS = 3
WILDCARD_MIN_PROBLEMS = 4
PROBLEM_MARKERS = ("Unavailable", "Doubtful")


def squad_exposure(counts: pd.DataFrame, squad_ids: list[int],
                   team_by_player: dict[int, int]) -> dict[int, dict]:
    """{event: {"doubles": n, "blanks": n}} over every event in `counts`.

    How many of your fifteen play twice in that gameweek, and how many not at
    all. This is fixture STRUCTURE only -- it carries no points estimate, which
    is exactly why it can be computed for gameweeks far past the xP horizon.
    """
    out: dict[int, dict] = {}
    for event, grp in counts.groupby("event"):
        n_by_team = {int(r["team_id"]): int(r["n_fixtures"]) for _, r in grp.iterrows()}
        fixtures = [n_by_team.get(team_by_player.get(int(p), -1), 1) for p in squad_ids]
        out[int(event)] = {
            "doubles": sum(1 for n in fixtures if n >= 2),
            "blanks": sum(1 for n in fixtures if n == 0),
        }
    return out


def captain_value_by_event(xp_df: pd.DataFrame, squad_ids: list[int]) -> dict[int, float]:
    """{event: xP of the best armband in the squad that week}.

    The captain is re-chosen every gameweek, so a Triple Captain week is worth
    the BEST player available that week, not whoever wears it today.
    """
    frame = xp_df.set_index("player_id")
    ids = [int(i) for i in squad_ids]
    return {event: float(frame.loc[ids, col].max())
            for event, col in event_columns(xp_df)}


BENCH_SIZE = 4


def bench_value_by_event(xp_df: pd.DataFrame, squad_ids: list[int]) -> dict[int, float]:
    """{event: xP a Bench Boost would add that week}.

    Approximated as the four LOWEST projections in the squad that gameweek.
    The real bench is chosen by the lineup optimizer under formation
    constraints, so this can differ by a fraction of a point -- it is an
    honest approximation and the advice text says so.
    """
    frame = xp_df.set_index("player_id")
    ids = [int(i) for i in squad_ids]
    return {event: float(frame.loc[ids, col].nsmallest(BENCH_SIZE).sum())
            for event, col in event_columns(xp_df)}


# How much a perfect structural week beyond the horizon can raise the bar on
# playing a chip now. 1.0 means a full-squad double doubles the threshold.
MAX_BAR_LIFT = 1.0
# Patience is worth nothing at the end of the season: an unplayed chip scores
# zero, so the lift fades to nothing over the last PATIENCE_WEEKS gameweeks.
PATIENCE_WEEKS = 10


def patience_bar(exposure: dict[int, dict], from_event: int, last_event: int,
                 horizon_last: int, kind: str, squad_size: int) -> tuple[float, int | None]:
    """How much harder to please the advisor should be, and which week it waits for.

    Only gameweeks PAST the xP horizon feed this. Inside the horizon the
    advisor compares real projections against each other and needs no proxy;
    past it there are no projections at all, so squad `exposure` to doubles or
    blanks is the only evidence -- and it may raise the bar, never veto.
    """
    beyond = {e: v[kind] for e, v in exposure.items()
              if horizon_last < e <= last_event and v[kind] > 0}
    if not beyond or squad_size <= 0:
        return 1.0, None
    target = max(beyond, key=lambda e: (beyond[e], -e))
    share = min(1.0, beyond[target] / float(squad_size))
    urgency = min(1.0, max(0, last_event - from_event) / float(PATIENCE_WEEKS))
    return 1.0 + MAX_BAR_LIFT * share * urgency, target


@dataclass
class ChipAdvice:
    chip: str | None
    reason: str
    # The gameweek the advisor is saving a chip for, when it declined to play
    # one now because a better week is visible. None when nothing is being held.
    hold_until: int | None = None


def _timing(value_now: float, by_event: dict[int, float], from_event: int,
            horizon_last: int, exposure: dict[int, dict], kind: str,
            base: float, last_event: int, squad_size: int):
    """(fires_now, target_event, why) for one chip.

    `why` is "projection" when a gameweek inside the horizon simply projects
    better, "structure" when only a fixture count past the horizon held it
    back, and None when the chip fires or nothing is being waited for. The two
    are reported differently on purpose -- see advise_chips.

    Two tiers of evidence, deliberately never mixed. Inside the xP horizon the
    advisor compares this week's real projection against other weeks' real
    projections and simply takes the best. Beyond it there are no projections,
    so fixture structure can only RAISE the bar on playing now -- it never
    vetoes a week that is genuinely outstanding, and it fades to nothing as the
    season runs out (see patience_bar).
    """
    bar, target = patience_bar(exposure, from_event, last_event, horizon_last,
                               kind, squad_size)
    future = {e: v for e, v in by_event.items() if e > from_event}
    if future:
        best_event = max(future, key=lambda e: future[e])
        if future[best_event] > value_now:
            return False, best_event, "projection"
    if value_now < base * bar:
        return False, target, "structure" if target is not None else None
    return True, None, None


def advise_chips(xp_df: pd.DataFrame, lineup, squad_ids: list[int],
                 counts: pd.DataFrame, team_by_player: dict[int, int],
                 from_event: int, chips_used: list[str],
                 last_event: int = 38) -> ChipAdvice:
    df = xp_df.set_index("player_id")
    used = set(chips_used or [])
    ids = [int(i) for i in squad_ids]
    n_by_team = {
        int(r["team_id"]): int(r["n_fixtures"])
        for _, r in counts[counts["event"] == from_event].iterrows()
    }

    def fixtures_for(pid: int) -> int:
        return n_by_team.get(team_by_player.get(int(pid), -1), 1)

    blanks = sum(1 for pid in ids if fixtures_for(pid) == 0)
    problems = sum(
        1 for pid in ids
        if any(m in f for f in df.loc[pid, "flags"] for m in PROBLEM_MARKERS)
    )
    cap_fixtures = fixtures_for(lineup.captain)
    bench_xp = [float(df.loc[pid, "xp_next1"]) for pid in lineup.bench]

    # Per-gameweek values, where the projection actually reaches. A frame with
    # no xp_gw columns (older callers, and every unit test that predates this)
    # degrades to the single-week behaviour these checks always had.
    cap_by_event = captain_value_by_event(xp_df, ids)
    bench_by_event = bench_value_by_event(xp_df, ids)
    exposure = squad_exposure(counts, ids, team_by_player)
    horizon_last = max(cap_by_event) if cap_by_event else from_event
    cap_xp = cap_by_event.get(from_event, float(df.loc[lineup.captain, "xp_next1"]))
    bench_total = bench_by_event.get(from_event, sum(bench_xp))
    squad_size = len(ids)

    holds: list[tuple[str, int, str]] = []

    # Free Hit needs no projection tier at all: its value IS the blank count,
    # and blanks are known from the fixture list for the whole rest of the
    # season. A bigger blank week ahead simply outranks a smaller one now.
    future_blanks = {e: v["blanks"] for e, v in exposure.items()
                     if e > from_event and v["blanks"] > blanks}
    fh_target = (max(future_blanks, key=lambda e: (future_blanks[e], -e))
                 if future_blanks else None)
    free_hit_ok = blanks >= FREE_HIT_MIN_BLANKS and fh_target is None
    wildcard_ok = problems >= WILDCARD_MIN_PROBLEMS

    triple_now, triple_target, triple_why = _timing(
        cap_xp, cap_by_event, from_event, horizon_last, exposure, "doubles",
        TRIPLE_CAPTAIN_MIN_XP, last_event, squad_size)
    # A double gameweek for the armband is still reason enough on its own, so
    # long as no week the model can SEE beats it.
    triple_ok = (triple_now or cap_fixtures >= 2) and triple_target is None

    bench_now, bench_target, bench_why = _timing(
        bench_total, bench_by_event, from_event, horizon_last, exposure,
        "doubles", BENCH_BOOST_MIN_XP * BENCH_SIZE, last_event, squad_size)
    # The per-player floor stays: four players each worth starting is a very
    # different bench from three blanks and a haul, and the sum hides that.
    bench_ok = bool(bench_xp) and min(bench_xp) >= BENCH_BOOST_MIN_XP and bench_now

    if blanks >= FREE_HIT_MIN_BLANKS and fh_target is not None             and "freehit" not in used:
        holds.append(("Free Hit", fh_target, "blanks"))
    if not triple_ok and triple_target is not None and "triplecaptain" not in used:
        holds.append(("Triple Captain", triple_target, triple_why))
    if not bench_ok and bench_target is not None and "benchboost" not in used             and bool(bench_xp) and min(bench_xp) >= BENCH_BOOST_MIN_XP:
        holds.append(("Bench Boost", bench_target, bench_why))

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
            f"{df.loc[lineup.captain, 'web_name']} has {detail} (xP {cap_xp:.1f}), and no "
            f"better armband week is visible through GW{horizon_last}. "
            f"Triple Captain turns that into 3x, but a blank or an early substitution "
            f"wastes the chip entirely."
        ))

    if bench_ok and "benchboost" not in used:
        return ChipAdvice("benchboost", (
            f"All four bench players project at {min(bench_xp):.1f}+ xP "
            f"({bench_total:.1f} total), the best bench week visible through "
            f"GW{horizon_last}. Bench Boost banks that — though the lineup "
            f"optimizer picks the real bench, so the total may shift slightly."
        ))

    if holds:
        # The two hold reasons are NOT interchangeable and must never be
        # described in the same words: one is a comparison of real projections,
        # the other is a fixture count with no projection behind it at all.
        parts = []
        for chip, event, why in holds:
            if why == "blanks":
                b = exposure.get(event, {}).get("blanks", 0)
                parts.append(f"{chip} for GW{event}, when {b} of your 15 blank "
                             f"against {blanks} this week")
            elif why == "projection":
                parts.append(f"{chip} for GW{event}, which projects better than this week")
            else:
                d = exposure.get(event, {}).get("doubles", 0)
                parts.append(f"{chip} for GW{event}, a double gameweek for {d} of your 15 "
                             f"(fixture count only — that far out there is no projection)")
        return ChipAdvice(None, "Holding " + "; ".join(parts) + ".",
                          hold_until=holds[0][1])

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
