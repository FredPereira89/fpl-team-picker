"""Read-only HTTP client for the public FPL API.

Never calls any endpoint requiring a login/session cookie (e.g. my-team/),
and never issues a non-GET request.
"""
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from .cache import Cache
from .throttle import FetchCancelled, TokenBucket

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; fpl-team-picker/1.0)"}
FORBIDDEN = ("my-team",)
# `history_past` for a completed season is immutable, so these can cache hard.
HISTORY_TTL_H = 24 * 30
DEFAULT_FETCH_RATE = 5.0
MAX_RETRIES = 3
BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_S = 120.0
POLL_S = 0.25


class ServerBackoff(Exception):
    """The server requested a delay longer than this run will wait."""


class DataCoverageError(RuntimeError):
    """Too much of this run's player history is missing to optimise safely.

    Raised rather than warned because the failure is invisible downstream: a
    player whose summary could not be fetched is zeroed and routed to the same
    price prior as a genuine new signing, and the optimizer then returns a
    confident, legal team built on an estimate nothing supports.
    """


class FplClient:
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0,
                 session=None, fetch_workers: int = 4,
                 fetch_rate_per_s: float | None = None):
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
        if fetch_rate_per_s is None:
            fetch_rate_per_s = DEFAULT_FETCH_RATE if rate_limit_s else 0.0
        self.fetch_workers = max(1, int(fetch_workers))
        self._limiter = TokenBucket(fetch_rate_per_s)
        # Real requests sessions belong to one worker thread each. Test and
        # benchmark sessions are injected and are expected to be thread safe.
        self._shared_session = session is not None
        self._local = threading.local()
        self._worker_sessions: list[requests.Session] = []
        self._sessions_lock = threading.Lock()

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

    def _cached(self, slug: str, ttl: float, not_before=None,
                require_final_through=None):
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

    def _store(self, slug: str, payload) -> None:
        self.cache.put(slug, payload, meta=self.snapshot_meta or None)
        self.cache.prune(slug, keep=3)
        self._record_source(slug)

    def _fallback(self, slug: str):
        got = self.cache.newest(slug)
        if got is None:
            return None
        self.stale = True
        self._record_source(slug)
        return got[0]

    def _get(self, path: str, slug: str, ttl_hours: float | None = None,
             not_before=None, require_final_through=None):
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cached = self._cached(slug, ttl, not_before, require_final_through)
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
            fallback = self._fallback(slug)
            if fallback is None:
                raise
            return fallback
        self._store(slug, payload)
        return payload

    def _worker_session(self):
        if self._shared_session:
            return self.session
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
            with self._sessions_lock:
                self._worker_sessions.append(session)
        return session

    def _close_worker_sessions(self) -> None:
        with self._sessions_lock:
            sessions, self._worker_sessions = self._worker_sessions, []
        for session in sessions:
            session.close()

    @staticmethod
    def _retry_after(resp) -> float | None:
        """Parse Retry-After delay seconds or an HTTP date."""
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        if value is None:
            return None
        value = str(value).strip()
        try:
            return max(float(value), 0.0)
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)

    @staticmethod
    def _pause(seconds: float, stop: threading.Event) -> None:
        if stop.wait(seconds):
            raise FetchCancelled

    def _fetch_json(self, url: str, stop: threading.Event):
        """Perform HTTP and retry on a worker; cache state stays on the caller."""
        if any(part in url.lower() for part in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        session = self._worker_session()
        for attempt in range(MAX_RETRIES + 1):
            last = attempt == MAX_RETRIES
            self._limiter.acquire(stop)
            try:
                response = session.get(url, timeout=30)
            except (requests.ConnectionError, requests.Timeout):
                if last:
                    raise
                self._pause(BACKOFF_S[attempt], stop)
                continue
            status = getattr(response, "status_code", 200)
            if status == 429 or 500 <= status < 600:
                asked = self._retry_after(response)
                if asked is not None and asked > MAX_RETRY_AFTER_S:
                    # A queued worker can start before the caller collects
                    # this future, so stop the shared fetch here.
                    stop.set()
                    raise ServerBackoff(f"{url}: Retry-After {asked:.0f}s")
                delay = max(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)], asked or 0.0)
                if status == 429:
                    self._limiter.pause_for(delay)
                if not last:
                    if status != 429:
                        self._pause(delay, stop)
                    continue
            response.raise_for_status()
            return response.json()

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
        ids = list(dict.fromkeys(int(pid) for pid in player_ids))
        found: dict[int, dict] = {}
        done = 0

        def tick() -> None:
            nonlocal done
            done += 1
            if progress:
                progress(done, len(ids))

        def fail(pid: int) -> None:
            self.stale = True
            self.fetch_failures.add(pid)

        def settle(pid: int, payload, fetched: bool) -> None:
            slug = f"element-summary-{pid}"
            try:
                if payload is None:
                    payload = self._fallback(slug)
                elif fetched:
                    self._store(slug, payload)
            except Exception:
                # A cache failure affects this player only. A successful fetch
                # whose write fails is not rescued by an older snapshot.
                payload = None
            if payload is None:
                fail(pid)
            else:
                found[pid] = payload
            tick()

        misses = []
        for pid in ids:
            try:
                cached = self._cached(f"element-summary-{pid}", ttl_hours,
                                      not_before, require_final_through)
            except Exception:
                # The sequential client marked a corrupt fresh snapshot failed
                # without attempting a replacement fetch.
                fail(pid)
                tick()
                continue
            if cached is None:
                misses.append(pid)
            else:
                found[pid] = cached
                tick()

        if misses:
            self._fetch_misses(misses, settle)
        return {pid: found[pid] for pid in ids if pid in found}

    def _fetch_misses(self, misses: list[int], settle) -> None:
        """Keep a bounded queue and integrate completed requests on this thread."""
        stop = threading.Event()
        remaining = iter(misses)
        pending: dict = {}
        pool = ThreadPoolExecutor(max_workers=self.fetch_workers)

        def submit_next() -> None:
            pid = next(remaining, None)
            if pid is not None:
                url = f"{BASE}element-summary/{pid}/"
                pending[pool.submit(self._fetch_json, url, stop)] = pid

        try:
            for _ in range(2 * self.fetch_workers):
                submit_next()
            while pending:
                finished, _ = wait(pending, timeout=POLL_S,
                                   return_when=FIRST_COMPLETED)
                for future in finished:
                    pid = pending.pop(future)
                    try:
                        payload = future.result()
                    except ServerBackoff:
                        stop.set()
                        payload = None
                    except Exception:
                        payload = None
                    settle(pid, payload, fetched=payload is not None)
                    if not stop.is_set():
                        submit_next()
            for pid in remaining:
                settle(pid, None, fetched=False)
        except BaseException:
            stop.set()
            raise
        finally:
            # Cancellation cannot interrupt an in-flight HTTP call. Join its
            # worker before closing that worker's session.
            pool.shutdown(wait=True, cancel_futures=True)
            self._close_worker_sessions()

    def entry(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/", f"entry-{entry_id}")

    def entry_history(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/history/", f"entry-history-{entry_id}")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/", f"entry-picks-{entry_id}-{gw}")
