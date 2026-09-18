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

Captures are VERSIONED, never overwritten. An earlier design kept only the first
capture per gameweek, on the theory that later runs carry hindsight -- but a
later PRE-deadline run carries legitimate team news, and it is the run the
manager actually acted on. So every capture is kept under its own timestamp,
the forecast manifest records which one a forecast read, and a replay selects
the capture its actioned forecast used, else the newest one strictly before the
deadline. `is_point_in_time` is False unless the selected capture demonstrably
PREDATES the deadline: a snapshot taken after kickoff records something, but
not what the manager knew.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

SNAPSHOT_DIR = "snapshots"
VERSION_FMT = "%Y%m%dT%H%M%S%fZ"


def _gw_dir(root, gw: int) -> Path:
    return Path(root) / SNAPSHOT_DIR / f"gw{int(gw)}"


def _instant(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _before(a, b) -> bool:
    x, y = _instant(a), _instant(b)
    return x is not None and y is not None and x < y


def capture(root, gw: int, *, bootstrap: dict, fixtures, deadline: str | None,
            sources: dict | None = None, final_through: int | None = None,
            captured_at=None) -> str:
    """Record the payloads this run read as a new version for `gw`.

    Returns the version id, which the caller should write into the forecast
    manifest so the forecast and the data it read stay tied together. Element
    summaries are deliberately NOT copied: they are already stored per round
    and cut exactly, so only their source stamps are recorded.
    """
    when = captured_at or datetime.now(timezone.utc)
    version = when.strftime(VERSION_FMT)
    out = _gw_dir(root, gw) / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap.json").write_text(json.dumps(bootstrap), encoding="utf-8")
    (out / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "gw": int(gw),
        "version": version,
        "captured_at": when.isoformat(),
        "deadline": deadline,
        "final_through": final_through,
        "sources": sources or {},
    }, sort_keys=True), encoding="utf-8")
    return version


def versions(root, gw: int) -> list[dict]:
    """Every capture's meta for `gw`, oldest first."""
    base = _gw_dir(root, gw)
    if not base.exists():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        meta = d / "meta.json"
        if meta.exists():
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return out


def select(root, gw: int, deadline: str | None = None,
           version: str | None = None) -> dict | None:
    """Which capture a replay of `gw` should read: meta, or None.

    A named version wins. Otherwise the newest capture strictly before the
    deadline -- the one that knew the most a manager could legitimately know --
    and failing that the newest at all, which `is_point_in_time` will then
    report as unusable evidence.
    """
    metas = versions(root, gw)
    if not metas:
        return None
    if version is not None:
        return next((m for m in reversed(metas) if m.get("version") == version), None)
    cutoff = deadline or next((m.get("deadline") for m in reversed(metas)
                               if m.get("deadline")), None)
    if cutoff:
        before = [m for m in metas if _before(m.get("captured_at"), cutoff)]
        if before:
            return before[-1]
    return metas[-1]


def load(root, gw: int, deadline: str | None = None,
         version: str | None = None) -> dict | None:
    """{"bootstrap":…, "fixtures":…, "meta":…} for the selected capture, or None."""
    meta = select(root, gw, deadline=deadline, version=version)
    if meta is None:
        return None
    d = _gw_dir(root, gw) / meta["version"]
    return {
        "bootstrap": json.loads((d / "bootstrap.json").read_text(encoding="utf-8")),
        "fixtures": json.loads((d / "fixtures.json").read_text(encoding="utf-8")),
        "meta": meta,
    }


def available(root) -> list[int]:
    base = Path(root) / SNAPSHOT_DIR
    if not base.exists():
        return []
    return sorted(int(d.name[2:]) for d in base.glob("gw*")
                  if d.name[2:].isdigit() and versions(root, int(d.name[2:])))


def is_point_in_time(root, gw: int, deadline: str | None = None,
                     version: str | None = None) -> bool:
    """True only when the selected capture was taken BEFORE its own deadline."""
    meta = select(root, gw, deadline=deadline, version=version)
    if meta is None:
        return False
    cutoff = deadline or meta.get("deadline")
    return _before(meta.get("captured_at"), cutoff)


def contamination_note(root, gws, versions_by_gw: dict | None = None) -> str | None:
    """A banner for any replay whose inputs are not point-in-time, or None.

    Returned rather than logged because the caller decides where it goes, and
    because a replay that cannot produce this note is the only kind whose edge
    can be quoted without a caveat.
    """
    chosen = versions_by_gw or {}
    missing = [int(g) for g in gws
               if not is_point_in_time(root, int(g), version=chosen.get(int(g)))]
    if not missing:
        return None
    listed = ", ".join(f"GW{g}" for g in missing)
    return (
        f"CONTAMINATED REPLAY — no pre-deadline snapshot exists for {listed}, so "
        f"today's prices, availability, news and club assignments are being used "
        f"for those gameweeks. The replay therefore knows about injuries and "
        f"transfers the live model could not, and any edge it reports is an "
        f"UPPER BOUND, not a measurement. Snapshots are captured on every run, "
        f"so future gameweeks will not carry this caveat."
    )
