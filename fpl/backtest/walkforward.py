"""Replay past gameweeks using only the data that existed before each one.

The prediction ledger held ONE scored gameweek. A single gameweek cannot tell
a good model from a lucky one: the weekly edge over the field has a standard
deviation near 15 points, so three gameweeks gave a 95% CI of [-36, +39] --
an interval that contains both "twice the edge needed to win the game" and
"far worse than average". Every other question in the 2026-09-09 statistical
review was blocked on that sample size.

Waiting for live gameweeks costs a season per 38 observations. Replaying them
costs minutes, because `element-summary` history is stored per round: asking
for the rounds strictly before gameweek N reconstructs exactly what was known
at its deadline.

Two contaminations survive and are worth stating plainly, because both flatter
the model rather than the reverse:

* `bootstrap-static` carries TODAY's price, status and news, not the deadline's.
  A player injured in September is marked unavailable in a replayed August
  gameweek, so the replay knows about absences the live model could not.
* Replaying rebuilds the squad from scratch each week -- a free wildcard every
  gameweek -- while a real season allows one transfer.

A replayed edge is therefore an UPPER BOUND on the live edge.
"""
import numpy as np
import pandas as pd
from scipy import stats

from ..data.normalize import history_current_frame, history_rounds_frame
from ..optimize.lineup import bench_order
from ..optimize.squad import XI_MIN

# Power to detect an edge, and the two-sided alpha it is detected at. Used to
# report what a given number of gameweeks could actually have shown.
POWER = 0.80
ALPHA = 0.05
# Gameweeks below which no verdict is offered at all.
MIN_GAMEWEEKS = 10


def forecast_inputs(summaries: dict, before_event: int) -> dict:
    """The season-to-date inputs as they stood before `before_event`.

    The single property this harness rests on: nothing from gameweek
    `before_event` or later may appear here. `element-summary` history is
    stored per round, so the cut is exact rather than approximate -- which is
    what separates a replay from scoring the model on its own answers.
    """
    return {
        "current": history_current_frame(summaries, before_event=int(before_event)),
        "rounds": history_rounds_frame(summaries, before_event=int(before_event)),
    }


def replayable_gameweeks(played, root, overwrite: bool = False) -> list[int]:
    """Which gameweeks a replay may write, leaving live forecasts alone.

    A replayed forecast is not the same object as a live one: it is built from
    TODAY's price, status and news rather than the deadline's, so it knows
    about absences the live model could not. Overwriting the pre-deadline
    record with it destroys the only uncontaminated evidence in the ledger --
    and that record is exactly what a scored gameweek is supposed to be.
    """
    from .ledger import available_gameweeks
    existing = set(available_gameweeks(root))
    return [int(gw) for gw in played
            if overwrite or int(gw) not in existing]


def actuals_frame(summaries: dict) -> pd.DataFrame:
    """Every recorded round's points and minutes, per player.

    Rows are summed within a round so a double gameweek counts both fixtures.
    """
    rows = []
    for pid, summary in (summaries or {}).items():
        for h in summary.get("history", []):
            rows.append({"player_id": int(pid), "round": int(h.get("round", -1)),
                         "actual": float(h.get("total_points", 0)),
                         "minutes": float(h.get("minutes", 0))})
    if not rows:
        return pd.DataFrame(columns=["player_id", "round", "actual", "minutes"])
    return pd.DataFrame(rows).groupby(["player_id", "round"], as_index=False).sum()


def autosub(squad_ids, starting_ids, frame) -> tuple[list[int], list[tuple[int, int]]]:
    """Apply FPL's automatic substitutions. Returns (final XI, [(out, in)]).

    A backtest that skips this charges the model for blanks a real manager
    would never have taken: FPL replaces any starter who did not appear with
    the first bench player who did, provided the formation survives.
    """
    pos = frame["position"].to_dict()
    mins = frame["minutes"].to_dict()
    xp = frame["xp_next1"].to_dict()
    xi = [int(p) for p in starting_ids]
    bench_ids = [int(p) for p in squad_ids if int(p) not in set(xi)]
    bench = bench_order(bench_ids, pos, xp)

    subs = []
    for out in [p for p in list(xi) if float(mins.get(p, 0)) <= 0]:
        for cand in list(bench):
            if float(mins.get(cand, 0)) <= 0:
                continue
            trial = [p for p in xi if p != out] + [cand]
            counts = pd.Series([pos[p] for p in trial]).value_counts()
            if counts.get("GKP", 0) != 1:
                continue
            if any(counts.get(k, 0) < v for k, v in XI_MIN.items()):
                continue
            xi, subs = trial, subs + [(out, cand)]
            bench.remove(cand)
            break
    return xi, subs


def realised_score(squad_ids, starting_ids, frame, captain=None, vice=None) -> dict:
    """What this squad ACTUALLY scored, with autosubs and the armband applied.

    `captain` and `vice` are the armband as it was actually set. When they are
    given they are honoured exactly: the captain scores double if he appeared,
    otherwise the vice does. When they are not given -- older callers, and the
    oracle -- the two highest projections in the deadline XI stand in, which is
    the old behaviour and NOT the production decision: the live lineup chooses
    the captain with `choose_captain`, which discounts a captain for his chance
    of not appearing, so it can differ from the raw top projection. A replay
    that reselected the armband was scoring a different decision from the one
    the tool recommends.

    Either way the pair is fixed at the deadline: a player autosubbed IN can
    never wear the armband, however well he then did.
    """
    xi, subs = autosub(squad_ids, starting_ids, frame)
    mins = frame["minutes"].to_dict()
    if captain is None:
        xp = frame["xp_next1"].to_dict()
        order = sorted([int(p) for p in starting_ids], key=lambda p: (-float(xp[p]), p))
        captain = order[0]
        vice = order[1] if len(order) > 1 else captain
    captain, vice = int(captain), int(vice if vice is not None else captain)
    armband = captain
    if float(mins.get(captain, 0)) <= 0 and float(mins.get(vice, 0)) > 0:
        armband = vice
    points = float(sum(float(frame.loc[p, "actual"]) for p in xi))
    return {"points": points + float(frame.loc[armband, "actual"]),
            "xi": xi, "subs": subs, "captain": armband,
            "named_captain": captain, "vice": vice}


def squad_ledger(by_gw: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """{gw: (our points, the field's average)} -> a frame with the edge."""
    rows = [{"gw": int(gw), "points": float(p), "field_average": float(a),
             "edge": float(p) - float(a)} for gw, (p, a) in sorted(by_gw.items())]
    return pd.DataFrame(rows)


def detectable_edge(n: int, sd: float) -> float:
    """The smallest weekly edge `n` gameweeks could confirm at 80% power.

    This is the number that decides whether a backtest result means anything.
    A full season resolves roughly 6.8 points a week at the observed SD of 15;
    confirming a 5-point edge needs about 71 gameweeks, or two seasons.
    """
    if n < 2:
        return float("inf")
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    return float(z * float(sd) / np.sqrt(n))


def weekly_edge(edges) -> dict:
    """Mean weekly edge with an interval, and an honest verdict about it.

    Reporting a bare mean from a handful of gameweeks is the single most
    misleading thing this backtest could do, so the interval and the
    minimum detectable effect are returned alongside it and the verdict
    refuses to call a sample too small to call.
    """
    e = np.asarray(list(edges), dtype=float)
    n = len(e)
    mean = float(e.mean()) if n else 0.0
    sd = float(e.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        t = stats.ttest_1samp(e, 0.0)
        tstat, pval = float(t.statistic), float(t.pvalue)
    else:
        lo = hi = float("nan")
        tstat = pval = float("nan")
    mde = detectable_edge(n, sd if n > 1 else 15.0)

    if n < MIN_GAMEWEEKS:
        verdict = (
            f"{n} gameweeks cannot settle this. At an SD of {sd:.1f} pts a week "
            f"the smallest edge {n} gameweeks could confirm is {mde:+.1f} pts/GW, "
            f"so any mean smaller than that is indistinguishable from luck. "
            f"Score more gameweeks before acting on this number."
        )
    elif pval < ALPHA and mean > 0:
        verdict = (f"Edge of {mean:+.1f} pts/GW confirmed over {n} gameweeks "
                   f"(p = {pval:.3f}).")
    else:
        verdict = (
            f"No edge demonstrated over {n} gameweeks: {mean:+.1f} pts/GW, "
            f"95% CI [{lo:+.1f}, {hi:+.1f}], p = {pval:.3f}. This sample could "
            f"only have detected {mde:+.1f} pts/GW or larger."
        )
    return {"n": n, "mean": mean, "sd": sd, "ci_low": lo, "ci_high": hi,
            "t": tstat, "p": pval, "detectable_edge": mde, "verdict": verdict}


def gameweek_inputs(root, gw: int, current_bootstrap: dict, current_fixtures,
                    summaries: dict, deadline: str | None = None,
                    snapshot_version: str | None = None) -> dict:
    """The bootstrap-derived inputs a replay of `gw` should use.

    From the gameweek's point-in-time snapshot when one exists -- the prices,
    availability, news, club assignments and fixture list as they stood before
    the deadline -- and from the current cache otherwise, in which case
    `point_in_time` is False and the caller must keep saying so. The earlier
    version of the walk-forward script imported the snapshot module and then
    read only the current cache, so a valid snapshot silenced the contamination
    banner without changing a single input.

    Returns {"players", "teams", "fixtures", "point_in_time", "source"}.
    """
    from ..data import snapshots
    from ..data.normalize import (normalize_players, normalize_teams,
                                  normalize_fixtures, history_past_frame,
                                  apply_season_baseline, latest_season)

    snap = snapshots.load(root, int(gw), deadline=deadline, version=snapshot_version)
    pit = snapshots.is_point_in_time(root, int(gw), deadline=deadline,
                                     version=snapshot_version)
    if snap is not None and pit:
        bootstrap, raw_fixtures, source = snap["bootstrap"], snap["fixtures"], "snapshot"
    else:
        bootstrap, raw_fixtures, source = current_bootstrap, current_fixtures, "current cache"
        pit = False

    players = normalize_players(bootstrap)
    past = history_past_frame(summaries)
    players = apply_season_baseline(players, past, latest_season(past))
    return {
        "players": players,
        "teams": normalize_teams(bootstrap),
        "fixtures": normalize_fixtures(raw_fixtures),
        "point_in_time": bool(pit),
        "source": source,
        "events": bootstrap.get("events", []),
    }
