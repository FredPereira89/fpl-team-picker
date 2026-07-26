"""Raw JSON snapshot cache with freshness checks and retention."""
from datetime import datetime, timezone
from pathlib import Path
import json

TS_FMT = "%Y%m%dT%H%M%SZ"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC, treating naive datetimes as UTC (not local time)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_matchday(fixtures: list[dict], now: datetime) -> bool:
    """True if any fixture kicks off on the same UTC date as `now`."""
    today = _as_utc(now).date()
    for f in fixtures or []:
        ko = f.get("kickoff_time")
        if not ko:
            continue
        if datetime.fromisoformat(ko.replace("Z", "+00:00")).astimezone(timezone.utc).date() == today:
            return True
    return False


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, slug: str) -> list[Path]:
        return sorted(self.root.glob(f"{slug}_*.json"), reverse=True)

    def put(self, slug: str, payload, now: datetime | None = None) -> Path:
        ts = _as_utc(now or _now())
        p = self.root / f"{slug}_{ts.strftime(TS_FMT)}.json"
        p.write_text(json.dumps(payload))
        return p

    def newest(self, slug: str):
        paths = self._paths(slug)
        if not paths:
            return None
        p = paths[0]
        ts = datetime.strptime(p.stem.rsplit("_", 1)[1], TS_FMT).replace(tzinfo=timezone.utc)
        return json.loads(p.read_text()), ts

    def get_fresh(self, slug: str, ttl_hours: float, now: datetime | None = None):
        got = self.newest(slug)
        if got is None:
            return None
        payload, ts = got
        age_h = (_as_utc(now or _now()) - ts).total_seconds() / 3600
        return payload if age_h < ttl_hours else None

    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)
        removed = 0
        for p in paths[keep:]:
            p.unlink()
            removed += 1
        return removed
