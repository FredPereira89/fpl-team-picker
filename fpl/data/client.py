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


class FplClient:
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0, session=None):
        self.cache = cache
        self.ttl_hours = ttl_hours
        self.rate_limit_s = rate_limit_s
        self.session = session or requests.Session()
        self.stale = False
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.rate_limit_s:
            delta = time.monotonic() - self._last_call
            if delta < self.rate_limit_s:
                time.sleep(self.rate_limit_s - delta)
            self._last_call = time.monotonic()

    def _get(self, path: str, slug: str, ttl_hours: float | None = None):
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cached = self.cache.get_fresh(slug, ttl)
        if cached is not None:
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
            return fallback[0]
        self.cache.put(slug, payload)
        self.cache.prune(slug, keep=3)
        return payload

    def bootstrap(self) -> dict:
        return self._get("bootstrap-static/", "bootstrap-static")

    def fixtures(self) -> list[dict]:
        return self._get("fixtures/", "fixtures")

    def element_summary(self, player_id: int, ttl_hours: float | None = None) -> dict:
        return self._get(f"element-summary/{player_id}/", f"element-summary-{player_id}",
                         ttl_hours=ttl_hours)

    def element_summaries(self, player_ids, ttl_hours: float = HISTORY_TTL_H,
                          progress=None) -> dict[int, dict]:
        """Fetch many element-summaries, tolerating individual failures.

        Defaults to a long TTL because the only field the weekly pipeline reads
        from these is `history_past`, which never changes once a season is over.
        A player whose summary can't be fetched is simply omitted; downstream
        that zeroes their baseline and routes them to the price prior.
        """
        out: dict[int, dict] = {}
        ids = list(player_ids)
        for i, pid in enumerate(ids):
            try:
                out[int(pid)] = self.element_summary(int(pid), ttl_hours=ttl_hours)
            except Exception:
                self.stale = True
            if progress:
                progress(i + 1, len(ids))
        return out

    def entry(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/", f"entry-{entry_id}")

    def entry_history(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/history/", f"entry-history-{entry_id}")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/", f"entry-picks-{entry_id}-{gw}")
