"""Final user-facing gameweek report.

Presentation only - never recomputes anything. Phrasing is always a
recommendation for the user to apply manually in the FPL app.
"""
from dataclasses import dataclass, field
import pandas as pd

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


@dataclass
class Recommendation:
    gw: int
    deadline: str
    mode: int
    lineup: object
    squad_ids: list[int]
    transfers: object | None = None
    chip: object | None = None
    bank: float = 0.0
    squad_value: float = 0.0
    flags: list[str] = field(default_factory=list)
    stale: bool = False
    trust: str = ""
    # Where this squad lands against a simulated field (optimize.rank), or None
    # when the distributional layer is switched off. Expected points alone say
    # nothing about rank; this is the number that answers the actual question.
    rank: dict | None = None
    # What the per-position recalibration was fitted on, or None when there is
    # not yet enough scored history to fit one.
    calibration: str | None = None
    # True when the fifteen below are the squad a CHIP builds rather than the
    # ordinary transfer plan, and true again when FPL will take them back at
    # the next deadline (Free Hit). The report has to say which, because
    # confirming one writes permanent state and the other must not.
    chip_squad: bool = False
    chip_temporary: bool = False


def _name(df, pid) -> str:
    row = df.loc[pid]
    return f"{row['web_name']} ({row['team']}, £{row['price']}m, {row['xp_next1']:.1f} xP)"


def _pair_by_position(df, out_ids, in_ids) -> list[tuple[int, int]]:
    """Match each departing player with the arriving one who replaces him.

    TransferPlan holds two sets, each sorted by player_id, and zipping them
    paired players at random: a two-move plan printed "Raya → Mbeumo", a keeper
    swapped for a midfielder, which FPL forbids and which made a perfectly legal
    squad look broken. An FPL squad is fixed at 2/5/5/3, so a transfer never
    changes the position counts and every player leaving has a counterpart
    arriving in his own position.
    """
    by_pos: dict[str, list[int]] = {}
    for i in in_ids:
        by_pos.setdefault(str(df.loc[i, "position"]), []).append(i)
    pairs, leftover = [], []
    for o in out_ids:
        same = by_pos.get(str(df.loc[o, "position"]))
        if same:
            pairs.append((o, same.pop(0)))
        else:
            leftover.append(o)
    # Positions always balance for a real plan; pair anything left over in order
    # rather than dropping a transfer silently from the report.
    rest = [i for ids in by_pos.values() for i in ids]
    pairs.extend(zip(leftover, rest))
    return pairs


def render(rec: Recommendation, xp_df: pd.DataFrame) -> str:
    df = xp_df.set_index("player_id")
    lu = rec.lineup
    out = [f"## Gameweek {rec.gw} — {rec.deadline}", ""]

    if rec.stale:
        out += ["> ⚠️ Live data was unavailable — this uses the most recent cached "
                "snapshot and may be stale.", ""]

    out += [f"### Starting XI ({lu.formation})"]
    for pos in POSITION_ORDER:
        members = [p for p in lu.xi if df.loc[p, "position"] == pos]
        if members:
            members.sort(key=lambda p: float(df.loc[p, "xp_next1"]), reverse=True)
            out.append(f"{pos}: " + ", ".join(_name(df, p) for p in members))
    out.append("")

    if rec.rank:
        r = rec.rank
        bar = ("the median manager" if abs(float(r["target"]) - 0.5) < 1e-9
               else f"the top {(1 - float(r['target'])) * 100:.0f}% of managers")
        # Whichever objective actually decided, this paragraph is always a
        # true statement about how the REPORTED team fares against the field
        # -- rank_stats is scored on the exact same XI now (C6-2). Only the
        # SELECTION sentence that follows has to say which objective chose
        # it: with expected points deciding by default, "chosen by how often
        # it beat the field" would be false.
        decided_by_rank = r.get("decided_by", "rank") == "rank"
        selection = (
            f"Chosen from {r['n_candidates']} candidate squads by how often "
            f"each beat a simulated field, not by expected points alone — "
            f"points your rivals also score do not move your rank."
            if decided_by_rank else
            f"Chosen from {r['n_candidates']} candidate squads by discounted "
            f"expected points over the horizon; the figures above are how "
            f"that choice happens to fare against a simulated field, not what "
            f"selected it."
        )
        out += [
            "### Against the field",
            f"Projected {r['mean_points']:.1f} points (± {r['sd_points']:.1f}), "
            f"which beats {bar} in **{r['p_beat_target']:.0%}** of simulated "
            f"gameweeks and finishes ahead of "
            f"**{r['rank_percentile']:.0%}** of rival squads on average.",
            "",
            selection,
            "",
            "> These percentages are **relative, not a forecast**. Both your "
            "squad and the simulated field are drawn from this model's own "
            "projections, so any optimism in them appears on both sides. "
            "Replayed against real GW1–3 results the simulation implied "
            "+15 pts/GW where the realised edge was +1.3. Use them to compare "
            "candidate squads, not to predict your finishing rank.",
            "",
        ]
        # `_honour_rank_captain` rejects the rank-preferred captain when he is
        # not in the exact weekly XI -- a real, silent divergence between what
        # rank chose and what is fielded, and it was recorded but never shown.
        if decided_by_rank and r.get("captain_reported") is False:
            out += [
                "> The captain rank would have preferred is not in this week's "
                "exact starting XI, so the armband below is the lineup's own "
                "choice instead.",
                "",
            ]

    out.append("### Bench (in order)")
    for n, pid in enumerate(lu.bench, start=1):
        note = " — reserve keeper" if df.loc[pid, "position"] == "GKP" else ""
        out.append(f"{n}. {_name(df, pid)}{note}")
    out.append("")

    cap, vice = df.loc[lu.captain], df.loc[lu.vice]
    cap_xp, vice_xp = float(cap["xp_next1"]), float(vice["xp_next1"])
    if cap_xp > vice_xp:
        comparison = (f"has the highest projected return in the squad "
                     f"({cap_xp:.1f} xP vs {vice_xp:.1f} for {vice['web_name']})")
    elif cap_xp == vice_xp:
        comparison = (f"is tied for the highest projected return in the squad "
                     f"({cap_xp:.1f} xP, matching {vice['web_name']})")
    else:
        # The armband falls to the vice if the captain does not appear at all,
        # so a captain with real rotation or fitness doubt over a strong vice
        # can be worth more than the higher projection on its own.
        comparison = (f"is the selected captain (projected {cap_xp:.1f} xP); "
                     f"{vice['web_name']} projects higher at {vice_xp:.1f} xP but "
                     f"is the vice, so his score still doubles if {cap['web_name']} "
                     f"does not appear")
    out += [
        f"### Captain: {cap['web_name']} (C)  |  Vice: {vice['web_name']} (VC)",
        f"Recommended because {cap['web_name']} {comparison}.",
        "",
    ]

    out.append("### Transfers this week")
    t = rec.transfers
    if t is None or t.n_transfers == 0:
        # Naming the objective matters. Whether holding won because no plan
        # beat the field, or because no plan gained discounted points over
        # the horizon, are different claims -- and since B6/R1 the horizon
        # decides by default, with rank only reporting on the result.
        if rec.mode == 2 and rec.rank and rec.rank.get("decided_by", "rank") == "rank":
            out.append(
                f"No transfer recommended — of {rec.rank['n_candidates']} plans "
                f"considered, none beat the field more often than holding once "
                f"its points hit was charged against it."
            )
        elif t is not None and getattr(t, "strategy", "single-period") == "multi-period":
            out.append(
                "No transfer recommended this week — the rolling plan finds no "
                "advantage to moving now once future free transfers and hits "
                "are included."
            )
        else:
            out.append("No transfer recommended — the squad is already optimal "
                       "on projected points.")
    else:
        for o, i in _pair_by_position(df, t.out_ids, t.in_ids):
            out.append(f"{df.loc[o, 'web_name']} → {df.loc[i, 'web_name']}")
        hit = f" after a -{t.hit_cost} hit" if t.hit_cost else " (no hit — within your free transfers)"
        # The solver maximises a decayed horizon, so this figure is not a raw
        # points total -- say so rather than letting it read as one.
        if getattr(t, "strategy", "single-period") == "multi-period":
            out.append(
                f"Suggested net gain of {t.gain:.1f} xP versus waiting one week"
                f"{hit} (rolling, discounted horizon; future transfers, hits "
                f"and {t.terminal_value:.1f} xP of terminal free-transfer value "
                f"are included)."
            )
            future_events = ", ".join(
                f"GW{step['event']}" for step in t.future_plan
                if step["out_ids"] or step["in_ids"])
            if future_events:
                out.append(
                    f"The current path also moves in {future_events}; those "
                    "moves are contingent and will be re-optimised each week."
                )
        else:
            out.append(f"Suggested net gain of {t.gain:.1f} xP across the horizon{hit} "
                       f"(discounted — gains in later gameweeks count for less).")
        p_pos = (rec.rank or {}).get("p_gain_positive")
        if p_pos is not None:
            # The expected gain is the decision; this is how sure the model is
            # of it THIS week under its own appearance and points scenarios,
            # including the reported captain/vice and legal autosubs. It does
            # not yet include epistemic parameter uncertainty (R4).
            out.append(f"In this week's scenarios the move comes out ahead of holding "
                       f"{p_pos:.0%} of the time (hit included).")
    out.append("")

    out.append("### Chip watch")
    if rec.chip is None:
        out.append("No chip recommended this week.")
    elif rec.chip.chip is None:
        # Saving a chip for a named week is a decision to act on, not the
        # absence of one -- it must not read like the "nothing doing" fallback.
        hold = rec.chip.hold_until
        prefix = f"**Hold until GW{hold}** — " if hold else ""
        out.append(prefix + rec.chip.reason)
    else:
        out.append(f"**{rec.chip.chip}** — {rec.chip.reason}")
        if rec.chip_temporary:
            out.append(
                "The fifteen above are this chip's ONE-WEEK squad. FPL restores "
                "your permanent team at the next deadline, and any money this "
                "week frees up does not carry over."
            )
        elif rec.chip_squad:
            out.append("The fifteen above are the squad this chip builds, "
                       "with every transfer free.")
    out.append("")

    out += [f"### Budget", f"Bank: £{rec.bank}m | Squad value: £{rec.squad_value}m", ""]

    out.append("### Flags")
    lines = []
    for pid in rec.squad_ids:
        for f in df.loc[pid, "flags"]:
            lines.append(f"- {df.loc[pid, 'web_name']}: {f}")
    lines += [f"- {f}" for f in rec.flags]
    if rec.trust:
        lines.append(f"- Model confidence: {rec.trust}")
    out += lines or ["- No outstanding injury or rotation concerns in this squad."]

    return "\n".join(out)
