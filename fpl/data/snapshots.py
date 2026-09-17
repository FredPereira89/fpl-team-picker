"""What was actually known at each gameweek's deadline.

The walk-forward harness reconstructs a past gameweek's forecast from
`element-summary` history, which is stored per round and therefore cuts
exactly. But it reads `bootstrap-static` as it stands TODAY, so the replay
knows today's price, status, news, team and set-piece order. A player injured
in September is marked unavailable in a replayed August gameweek; a player who
moved clubs is mapped to his later club. The replay knows about absences the
live model could not, and the resulting edge flatters the model.

The cache cannot fix this after the fact -- it keeps three snapshots per slug,
so the bootstrap that existed at GW1's deadline is long gone. What it can do is
stop the problem recurring: every run writes the payloads it read, keyed by the
gameweek it was planning, and a later replay can ask for those instead.

Two rules make the record trustworthy:

* the FIRST capture for a gameweek wins. A later run in the same gameweek has
  seen more football and more team news, so overwriting would replace the
  deadline's knowledge with hindsight -- exactly the failure this exists to
  prevent.
* `is_point_in_time` is False unless the capture demonstrably PREDATES the
  deadline. A snapshot taken after kickoff is a record of something, but not
  of what the manager knew.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

SNAPSHOT_DIR = "snapshots"


def _dir(root, gw: int) -> Path:
    return Path(root) / SNAPSHOT_DIR / f"gw{int(gw)}"


def capture(root, gw: int, *, bootstrap: dict, fixtures, deadline: str | None,
            sources: dict | None = None, final_through: int | None = None,
            captured_at=None) -> Path | None:
    """Record the payloads this run read, as the point-in-time set for `gw`.

    Returns the directory written, or None when a capture already exists --
    the first one is the one that describes the deadline, so this never
    overwrites. Element summaries are deliberately NOT copied: they are already
    stored per round and cut exactly, so only their source stamps are recorded.
    """
    out = _dir(root, gw)
    if (out / "meta.json").exists():
        return None
    out.mkdir(parents=True, exist_ok=True)
    when = captured_at or datetime.now(timezone.utc)

    (out / "bootstrap.json").write_text(json.dumps(bootstrap), encoding="utf-8")
    (out / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "gw": int(gw),
        "captured_at": when.isoformat() if hasattr(when, "isoformat") else str(when),
        "deadline": deadline,
        "final_through": final_through,
        "sources": sources or {},
    }, sort_keys=True), encoding="utf-8")
    return out


def load(root, gw: int) -> dict | None:
    """{"bootstrap":…, "fixtures":…, "meta":…} for `gw`, or None."""
    d = _dir(root, gw)
    if not (d / "meta.json").exists():
        return None
    return {
        "bootstrap": json.loads((d / "bootstrap.json").read_text(encoding="utf-8")),
        "fixtures": json.loads((d / "fixtures.json").read_text(encoding="utf-8")),
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
    }


def available(root) -> list[int]:
    base = Path(root) / SNAPSHOT_DIR
    if not base.exists():
        return []
    out = []
    for d in base.glob("gw*"):
        if (d / "meta.json").exists() and d.name[2:].isdigit():
            out.append(int(d.name[2:]))
    return sorted(out)


def is_point_in_time(root, gw: int) -> bool:
    """True only when `gw`'s snapshot was captured BEFORE its own deadline.

    A capture taken afterwards records what was known once the team sheets were
    out, which is not what the manager was deciding on -- so a replay built
    from it is still an upper bound, and must keep saying so.
    """
    snap = load(root, gw)
    if snap is None:
        return False
    meta = snap["meta"]
    captured, deadline = meta.get("captured_at"), meta.get("deadline")
    if not captured or not deadline:
        return False
    return str(captured) < str(deadline)


def contamination_note(root, gws) -> str | None:
    """A banner for any replay whose inputs are not point-in-time, or None.

    Returned rather than logged because the caller decides where it goes, and
    because a replay that cannot produce this note is the only kind whose edge
    can be quoted without a caveat.
    """
    missing = [int(g) for g in gws if not is_point_in_time(root, int(g))]
    if not missing:
        return None
    listed = ", ".join(f"GW{g}" for g in missing)
    return (
        f"CONTAMINATED REPLAY — no pre-deadline snapshot exists for {listed}, so "
        f"today's prices, availability, news and club assignments are being used "
        f"for those gameweeks. The replay therefore knows about injuries and "
        f"transfers the live model could not, and any edge it reports is an "
        f"UPPER BOUND, not a measurement. Snapshots are captured from now on, so "
        f"future gameweeks will not carry this caveat."
    )
