"""Raw JSON snapshot cache with freshness checks and retention."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

TS_FMT = "%Y%m%dT%H%M%SZ"
# Kickoff to the point a finished match's returns are on record. A match runs
# ~2h including stoppage; the extra hour is slack for FPL's own processing.
MATCH_COMPLETE_H = 3.0
# Kickoff to the point FPL's own bonus and stat check has normally landed.
# Only used to warn about snapshots taken before `final_through` was recorded.
FINAL_CHECK_H = 6.0
META_SUFFIX = ".meta"


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

    A match counts once FPL sets `finished_provisional`, at the final whistle.
    Waiting for `finished` -- which flips only after FPL's bonus and stat check,
    hours later -- would leave a window where the football has been played, the
    history rows exist, and a snapshot taken before kickoff still passes as
    current. GW3 2026/27 sat in that state: ten matches played, `finished`
    False on every one of them.

    None when no fixture has finished (pre-season), which leaves the ordinary
    TTL in charge.

    Note what this does NOT establish: that the data is FINAL. It answers "has
    the football been played", and FPL's bonus and stat check lands after that
    -- see `final_through` and `settled_after`.
    """
    kickoffs = []
    for f in fixtures or []:
        if not (f.get("finished") or f.get("finished_provisional")):
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


def final_through(fixtures: list[dict]) -> int:
    """Highest gameweek FPL has fully checked -- every fixture `finished`.

    `finished_provisional` flips at the final whistle, `finished` only after
    the bonus and stat review. A snapshot taken between the two carries numbers
    that are still moving, and no timestamp can tell you which side of the
    review it landed on -- so this is recorded WITH each snapshot (see
    `Cache.put(meta=...)`) rather than inferred from its age.
    """
    events: dict[int, bool] = {}
    for f in fixtures or []:
        ev = f.get("event")
        if ev is None:
            continue
        ev = int(ev)
        events[ev] = events.get(ev, True) and bool(f.get("finished"))
    checked = [e for e, done in events.items() if done]
    return max(checked) if checked else 0


def settled_after(fixtures: list[dict], gw: int) -> datetime | None:
    """When a gameweek's data can be assumed settled, for snapshots that
    predate the `final_through` marker and cannot say for themselves.

    A heuristic, and only ever used to WARN: FPL usually confirms bonus within
    a couple of hours of a gameweek's last whistle, but a stat correction can
    land later, and the only honest thing to do with an old snapshot is say it
    might have been taken mid-review.
    """
    kickoffs = [
        datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
        for f in fixtures or []
        if f.get("kickoff_time") and int(f.get("event") or 0) == int(gw)
    ]
    if not kickoffs:
        return None
    return max(kickoffs) + timedelta(hours=FINAL_CHECK_H)


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, slug: str) -> list[Path]:
        return sorted(self.root.glob(f"{slug}_*.json"), reverse=True)

    def put(self, slug: str, payload, now: datetime | None = None,
            meta: dict | None = None) -> Path:
        """Write a snapshot, and beside it what was true of the data when it was
        taken (`meta`), which no later reader can work out from the timestamp.

        The sidecar deliberately does not end in `.json`: `_paths` globs for
        snapshots and would otherwise try to read it as one.
        """
        ts = _as_utc(now or _now())
        p = self.root / f"{slug}_{ts.strftime(TS_FMT)}.json"
        p.write_text(json.dumps(payload))
        if meta:
            p.with_suffix(META_SUFFIX).write_text(json.dumps(meta))
        return p

    def newest(self, slug: str):
        paths = self._paths(slug)
        if not paths:
            return None
        p = paths[0]
        ts = datetime.strptime(p.stem.rsplit("_", 1)[1], TS_FMT).replace(tzinfo=timezone.utc)
        return json.loads(p.read_text()), ts

    def newest_stamp(self, slug: str) -> datetime | None:
        """When the newest snapshot was taken, without reading it.

        `newest` deserialises the whole payload, which is wasteful when all the
        caller wants is the age -- and for 650 element-summaries it is the
        difference between a few milliseconds and re-parsing tens of megabytes.
        """
        paths = self._paths(slug)
        if not paths:
            return None
        return datetime.strptime(paths[0].stem.rsplit("_", 1)[1], TS_FMT).replace(
            tzinfo=timezone.utc)

    def newest_meta(self, slug: str) -> dict:
        """What the newest snapshot recorded about itself, or {} if it is older
        than this mechanism (in which case nothing can be assumed about it)."""
        paths = self._paths(slug)
        if not paths:
            return {}
        sidecar = paths[0].with_suffix(META_SUFFIX)
        if not sidecar.exists():
            return {}
        try:
            return json.loads(sidecar.read_text()) or {}
        except (json.JSONDecodeError, OSError):
            return {}

    def get_fresh(self, slug: str, ttl_hours: float, now: datetime | None = None,
                  not_before: datetime | None = None,
                  require_final_through: int | None = None):
        """Cached payload if it is within its TTL, new enough to carry whatever
        `not_before` says it must (see `data_complete_after`), and -- if it says
        so for itself -- taken after FPL had checked gameweek
        `require_final_through`.

        A snapshot with no marker is accepted: it predates the mechanism, and
        refusing every one of them would force a full re-fetch of 650 players
        for data that is usually already settled. `settled_after` is what tells
        the caller to be suspicious of those.
        """
        got = self.newest(slug)
        if got is None:
            return None
        payload, ts = got
        if not_before is not None and ts < _as_utc(not_before):
            return None
        if require_final_through is not None:
            recorded = self.newest_meta(slug).get("final_through")
            if recorded is not None and int(recorded) < int(require_final_through):
                return None
        age_h = (_as_utc(now or _now()) - ts).total_seconds() / 3600
        return payload if age_h < ttl_hours else None

    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)
        removed = 0
        for p in paths[keep:]:
            p.unlink()
            p.with_suffix(META_SUFFIX).unlink(missing_ok=True)
            removed += 1
        return removed
