"""Raw JSON snapshot cache with freshness checks and retention."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

TS_FMT = "%Y%m%dT%H%M%SZ"
# Kickoff to the point a finished match's returns are on record. A match runs
# ~2h including stoppage; the extra hour is slack for FPL's own processing.
MATCH_COMPLETE_H = 3.0


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


def data_complete_after(fixtures: list[dict]) -> datetime | None:
    """The instant by which every finished match's returns are on record.

    A snapshot taken before this cannot contain the latest gameweek's history
    no matter how generous its TTL, which is the failure that kept the current
    season invisible to the model: element-summaries cache for 30 days, and
    since `history_current_frame` started reading this season's rounds out of
    them, "not yet expired" stopped meaning "still correct".

    None when no fixture has finished (pre-season), which leaves the ordinary
    TTL in charge.
    """
    kickoffs = []
    for f in fixtures or []:
        if not f.get("finished"):
            continue
        ko = f.get("kickoff_time")
        if not ko:
            continue
        kickoffs.append(
            datetime.fromisoformat(ko.replace("Z", "+00:00")).astimezone(timezone.utc)
        )
    if not kickoffs:
        return None
    return max(kickoffs) + timedelta(hours=MATCH_COMPLETE_H)


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

    def get_fresh(self, slug: str, ttl_hours: float, now: datetime | None = None,
                  not_before: datetime | None = None):
        """Cached payload if it is both within its TTL and new enough to carry
        whatever `not_before` says it must (see `data_complete_after`)."""
        got = self.newest(slug)
        if got is None:
            return None
        payload, ts = got
        if not_before is not None and ts < _as_utc(not_before):
            return None
        age_h = (_as_utc(now or _now()) - ts).total_seconds() / 3600
        return payload if age_h < ttl_hours else None

    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)
        removed = 0
        for p in paths[keep:]:
            p.unlink()
            removed += 1
        return removed
