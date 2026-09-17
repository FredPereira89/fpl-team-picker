"""Read-only HTTP client for the public FPL API.

Never calls any endpoint requiring a login/session cookie (e.g. my-team/),
and never issues a non-GET request.
"""
import time
import requests
from .cache import Cache

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; fpl-team-picker/1.0)"}
FORBIDDEN = ("my-team",)
# `history_past` for a completed season is immutable, so these can cache hard.
HISTORY_TTL_H = 24 * 30


class DataCoverageError(RuntimeError):
    """Too much of this run's player history is missing to optimise safely.

    Raised rather than warned because the failure is invisible downstream: a
    player whose summary could not be fetched is zeroed and routed to the same
    price prior as a genuine new signing, and the optimizer then returns a
    confident, legal team built on an estimate nothing supports.
    """


class FplClient:
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0, session=None):
        self.cache = cache
        self.ttl_hours = ttl_hours
        self.rate_limit_s = rate_limit_s
        self.session = session or requests.Session()
        self.stale = False
        # When each payload this run used was captured. A forecast is only
        # reproducible if it records which snapshots it read: "the model got
        # worse" and "the model was handed a week-old bootstrap" look identical
        # in the score ledger otherwise.
        self.sources: dict[str, str] = {}
        # Recorded alongside every snapshot this client writes. `final_through`
        # says which gameweeks FPL had finished CHECKING at capture time, which
        # a timestamp cannot express and a later reader cannot recover.
        self.snapshot_meta: dict = {}
        # Slugs served from a snapshot too old to carry a `final_through`
        # marker, when one was asked for.
        self.unverified: set[str] = set()
        # Players whose element-summary could not be fetched OR served from
        # cache on this run. `stale` is one global boolean and cannot say WHO
        # is affected, which is exactly what the caller needs to decide whether
        # the missing history touches the squad.
        self.fetch_failures: set[int] = set()
        self._last_call = 0.0

    def _record_source(self, slug: str) -> None:
        ts = self.cache.newest_stamp(slug)
        if ts is not None:
            self.sources[slug] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def source_summary(self) -> dict:
        """Snapshot times, with the per-player summaries collapsed to a range.

        Seven hundred element-summary entries say nothing a first-and-last pair
        does not, and would dwarf the forecast they annotate.
        """
        out, summaries = {}, []
        for slug, ts in self.sources.items():
            if slug.startswith("element-summary-"):
                summaries.append(ts)
            else:
                out[slug] = ts
        if summaries:
            out["element-summary"] = {"oldest": min(summaries), "newest": max(summaries),
                                      "n": len(summaries)}
        return out

    def _throttle(self) -> None:
        if self.rate_limit_s:
            delta = time.monotonic() - self._last_call
            if delta < self.rate_limit_s:
                time.sleep(self.rate_limit_s - delta)
            self._last_call = time.monotonic()

    def _get(self, path: str, slug: str, ttl_hours: float | None = None,
             not_before=None, require_final_through=None):
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cached = self.cache.get_fresh(slug, ttl, not_before=not_before,
                                      require_final_through=require_final_through)
        if cached is not None:
            self._record_source(slug)
            # Served a snapshot that predates the `final_through` marker, so it
            # cannot say whether it was taken before or after FPL's data check.
            # It is still used -- refusing every pre-marker snapshot would
            # re-fetch 650 players for data that is usually settled -- but the
            # caller is told, so a report can carry the caveat.
            if (require_final_through is not None
                    and "final_through" not in self.cache.newest_meta(slug)):
                self.unverified.add(slug)
            return cached
        url = BASE + path
        if any(f in url.lower() for f in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        try:
            self._throttle()
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            fallback = self.cache.newest(slug)
            if fallback is None:
                raise
            self.stale = True
            self._record_source(slug)
            return fallback[0]
        self.cache.put(slug, payload, meta=self.snapshot_meta or None)
        self.cache.prune(slug, keep=3)
        self._record_source(slug)
        return payload

    def bootstrap(self) -> dict:
        return self._get("bootstrap-static/", "bootstrap-static")

    def fixtures(self) -> list[dict]:
        return self._get("fixtures/", "fixtures")

    def element_summary(self, player_id: int, ttl_hours: float | None = None,
                        not_before=None, require_final_through=None) -> dict:
        return self._get(f"element-summary/{player_id}/", f"element-summary-{player_id}",
                         ttl_hours=ttl_hours, not_before=not_before,
                         require_final_through=require_final_through)

    def element_summaries(self, player_ids, ttl_hours: float = HISTORY_TTL_H,
                          progress=None, not_before=None,
                          require_final_through=None) -> dict[int, dict]:
        """Fetch many element-summaries, tolerating individual failures.

        The long default TTL dates from when `history_past` was the only field
        read from these — it is immutable once a season ends, so age did not
        matter. `history_current_frame` now reads THIS season's rounds from the
        same payload, which age very much does affect, so callers that need the
        current season must pass `not_before=data_complete_after(fixtures)`.
        Without it a 30-day-old snapshot counts as fresh and the model silently
        runs on whatever gameweek happened to be current when it was taken.

        A player whose summary can't be fetched is omitted from the result AND
        recorded in `fetch_failures`. Downstream, an omitted player is zeroed
        and routed to the price prior -- the right treatment for a newcomer and
        badly wrong for an established player lost to an outage -- so the
        caller has to be able to tell the two apart.
        """
        out: dict[int, dict] = {}
        ids = list(player_ids)
        for i, pid in enumerate(ids):
            try:
                out[int(pid)] = self.element_summary(
                    int(pid), ttl_hours=ttl_hours, not_before=not_before,
                    require_final_through=require_final_through)
            except Exception:
                self.stale = True
                self.fetch_failures.add(int(pid))
            if progress:
                progress(i + 1, len(ids))
        return out

    def entry(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/", f"entry-{entry_id}")

    def entry_history(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/history/", f"entry-history-{entry_id}")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/", f"entry-picks-{entry_id}-{gw}")
