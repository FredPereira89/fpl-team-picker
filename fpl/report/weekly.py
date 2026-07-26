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


def _name(df, pid) -> str:
    row = df.loc[pid]
    return f"{row['web_name']} ({row['team']}, £{row['price']}m, {row['xp_next1']:.1f} xP)"


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

    out.append("### Bench (in order)")
    for n, pid in enumerate(lu.bench, start=1):
        note = " — reserve keeper" if df.loc[pid, "position"] == "GKP" else ""
        out.append(f"{n}. {_name(df, pid)}{note}")
    out.append("")

    cap, vice = df.loc[lu.captain], df.loc[lu.vice]
    out += [
        f"### Captain: {cap['web_name']} (C)  |  Vice: {vice['web_name']} (VC)",
        f"Recommended because {cap['web_name']} has the highest projected return in the "
        f"squad ({cap['xp_next1']:.1f} xP vs {vice['xp_next1']:.1f} for {vice['web_name']}).",
        "",
    ]

    out.append("### Transfers this week")
    t = rec.transfers
    if t is None or t.n_transfers == 0:
        out.append("No transfer recommended — the squad is already optimal on projected points.")
    else:
        for o, i in zip(t.out_ids, t.in_ids):
            out.append(f"{df.loc[o, 'web_name']} → {df.loc[i, 'web_name']}")
        hit = f" after a -{t.hit_cost} hit" if t.hit_cost else " (no hit — within your free transfers)"
        out.append(f"Suggested net gain of {t.gain:.1f} xP over the next gameweeks{hit}.")
    out.append("")

    out.append("### Chip watch")
    if rec.chip is None or rec.chip.chip is None:
        out.append(rec.chip.reason if rec.chip else "No chip recommended this week.")
    else:
        out.append(f"**{rec.chip.chip}** — {rec.chip.reason}")
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
