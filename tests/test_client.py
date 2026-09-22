from datetime import datetime, timedelta, timezone

import pytest
from fpl.data.cache import Cache
from fpl.data.client import FplClient, BASE

def now() -> datetime:
    """Real current time: FplClient has no clock injection point, so these
    tests place cache entries relative to the wall clock rather than a fixed
    date that would drift past the TTL being exercised."""
    return datetime.now(timezone.utc)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes, fail=False):
        self.routes, self.fail, self.calls = routes, fail, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.fail:
            raise RuntimeError("network down")
        return FakeResponse(self.routes[url])


def test_bootstrap_fetches_and_caches(tmp_path):
    s = FakeSession({BASE + "bootstrap-static/": {"elements": []}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    assert c.bootstrap() == {"elements": []}
    assert c.stale is False
    assert len(s.calls) == 1


def test_second_call_uses_cache_not_network(tmp_path):
    s = FakeSession({BASE + "bootstrap-static/": {"elements": []}})
    cache = Cache(tmp_path)
    FplClient(cache, rate_limit_s=0, session=s).bootstrap()
    FplClient(cache, rate_limit_s=0, session=s).bootstrap()
    assert len(s.calls) == 1


def test_falls_back_to_stale_cache_on_failure(tmp_path):
    cache = Cache(tmp_path)
    cache.put("bootstrap-static", {"old": True})
    c = FplClient(cache, ttl_hours=0, rate_limit_s=0, session=FakeSession({}, fail=True))
    assert c.bootstrap() == {"old": True}
    assert c.stale is True


def test_raises_when_failure_and_no_cache(tmp_path):
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=FakeSession({}, fail=True))
    with pytest.raises(RuntimeError):
        c.bootstrap()


def test_never_requests_authenticated_endpoints(tmp_path):
    s = FakeSession({BASE + "fixtures/": [], BASE + "bootstrap-static/": {}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    c.fixtures()
    c.bootstrap()
    assert all("my-team" not in u for u in s.calls)


def test_entry_picks_url_shape(tmp_path):
    url = BASE + "entry/123/event/1/picks/"
    s = FakeSession({url: {"picks": []}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    assert c.entry_picks(123, 1) == {"picks": []}
    assert s.calls == [url]


def test_element_summary_refetches_when_cache_predates_last_gameweek(tmp_path):
    """The bug: element-summaries default to a 30-day TTL, but since the current
    season's `history` started being read from them, a cache older than the last
    completed gameweek is wrong however fresh its TTL says it is."""
    url = BASE + "element-summary/411/"
    s = FakeSession({url: {"history": [{"round": 1}, {"round": 2}, {"round": 3}]}})
    cache = Cache(tmp_path)
    cache.put("element-summary-411", {"history": [{"round": 1}]},
              now=now() - timedelta(days=11))
    c = FplClient(cache, rate_limit_s=0, session=s)

    got = c.element_summary(411, ttl_hours=720, not_before=now() - timedelta(days=2))

    assert [h["round"] for h in got["history"]] == [1, 2, 3]
    assert s.calls == [url]


def test_element_summary_keeps_cache_written_after_last_gameweek(tmp_path):
    url = BASE + "element-summary/411/"
    s = FakeSession({url: {"history": [{"round": 1}, {"round": 2}]}})
    cache = Cache(tmp_path)
    cache.put("element-summary-411", {"history": [{"round": 1}, {"round": 2}]},
              now=now() - timedelta(hours=1))
    c = FplClient(cache, rate_limit_s=0, session=s)

    c.element_summary(411, ttl_hours=720, not_before=now() - timedelta(days=2))

    assert s.calls == []  # still current — no refetch, no 10-minute crawl


def test_element_summaries_passes_not_before_through(tmp_path):
    urls = {BASE + f"element-summary/{i}/": {"history": [{"round": 2}]} for i in (1, 2)}
    s = FakeSession(urls)
    cache = Cache(tmp_path)
    for i in (1, 2):
        cache.put(f"element-summary-{i}", {"history": []},
                  now=now() - timedelta(days=11))
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2], not_before=now() - timedelta(days=2))

    assert len(s.calls) == 2
    assert all(o["history"] == [{"round": 2}] for o in out.values())


def test_forbids_my_team_exact_case(tmp_path):
    s = FakeSession({})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    with pytest.raises(ValueError, match="refusing to call authenticated endpoint"):
        c._get("entry/1/my-team/", "entry-myteam")
    assert len(s.calls) == 0  # No network call made


def test_forbids_my_team_case_variant(tmp_path):
    s = FakeSession({})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    with pytest.raises(ValueError, match="refusing to call authenticated endpoint"):
        c._get("entry/1/My-Team/", "entry-myteam")
    assert len(s.calls) == 0  # No network call made


# --- element_summaries semantics --------------------------------------------
# Pinned against the sequential implementation before the fetch went
# concurrent. They must pass, unchanged, on both.

class PerUrlSession:
    """Routes by URL; any URL in `failing` raises like a dropped connection."""

    def __init__(self, routes, failing=()):
        self.routes, self.failing, self.calls = routes, set(failing), []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.failing:
            raise RuntimeError("connection reset")
        return FakeResponse(self.routes[url])


def _es(pid):
    return BASE + f"element-summary/{pid}/"


def test_element_summaries_mixes_hits_and_fetches(tmp_path):
    cache = Cache(tmp_path)
    for pid in (1, 2):
        cache.put(f"element-summary-{pid}", {"id": pid, "src": "cache"},
                  now=now() - timedelta(hours=1))
    s = PerUrlSession({_es(3): {"id": 3, "src": "net"}, _es(4): {"id": 4, "src": "net"}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2, 3, 4])

    assert list(out) == [1, 2, 3, 4]
    assert [out[p]["src"] for p in (1, 2, 3, 4)] == ["cache", "cache", "net", "net"]
    assert sorted(s.calls) == [_es(3), _es(4)]
    assert c.stale is False and c.fetch_failures == set()
    assert set(c.sources) == {f"element-summary-{p}" for p in (1, 2, 3, 4)}
    assert cache.newest("element-summary-3")[0] == {"id": 3, "src": "net"}


def test_element_summaries_writes_the_snapshot_marker(tmp_path):
    s = PerUrlSession({_es(7): {"id": 7}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    c.snapshot_meta = {"final_through": 5}

    c.element_summaries([7])

    assert c.cache.newest_meta("element-summary-7") == {"final_through": 5}


def test_a_failed_fetch_falls_back_to_the_old_snapshot(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-5", {"id": 5, "src": "old"}, now=now() - timedelta(days=11))
    s = PerUrlSession({}, failing=[_es(5)])
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([5], not_before=now() - timedelta(days=2))

    assert out == {5: {"id": 5, "src": "old"}}
    assert c.stale is True
    assert c.fetch_failures == set()
    assert "element-summary-5" in c.sources


def test_a_failed_fetch_with_no_snapshot_is_recorded_and_omitted(tmp_path):
    s = PerUrlSession({_es(1): {"id": 1}}, failing=[_es(6)])
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    out = c.element_summaries([1, 6])

    assert list(out) == [1]
    assert c.stale is True
    assert c.fetch_failures == {6}


def test_a_snapshot_without_the_marker_is_flagged_unverified(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-1", {"id": 1}, now=now() - timedelta(hours=1))
    cache.put("element-summary-2", {"id": 2}, now=now() - timedelta(hours=1),
              meta={"final_through": 5})
    c = FplClient(cache, rate_limit_s=0, session=PerUrlSession({}))

    c.element_summaries([1, 2], require_final_through=5)

    assert c.unverified == {"element-summary-1"}


def test_progress_counts_every_player_once_in_order(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-1", {"id": 1}, now=now() - timedelta(hours=1))
    s = PerUrlSession({_es(p): {"id": p} for p in (2, 3, 4)}, failing=[_es(5)])
    c = FplClient(cache, rate_limit_s=0, session=s)
    seen = []

    c.element_summaries([1, 2, 3, 4, 5], progress=lambda d, t: seen.append((d, t)))

    assert [d for d, _ in seen] == [1, 2, 3, 4, 5]
    assert {t for _, t in seen} == {5}


# Per-player containment: the sequential loop wraps EVERYTHING for one player
# -- cache read, fetch, write, prune, fallback -- in one try/except, so a
# broken cache entry fails that player and the run moves on.

def _write_corrupt(cache, slug, age):
    from fpl.data.cache import TS_FMT
    stamp = (now() - age).strftime(TS_FMT)
    (cache.root / f"{slug}_{stamp}.json").write_text("{not json")


def test_a_corrupt_cached_snapshot_fails_only_that_player(tmp_path):
    cache = Cache(tmp_path)
    _write_corrupt(cache, "element-summary-1", timedelta(hours=1))   # fresh, unreadable
    s = PerUrlSession({_es(2): {"id": 2}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2])

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}
    assert c.stale is True
    assert s.calls == [_es(2)]            # today: marked failed, not refetched


def test_a_failed_cache_write_fails_only_that_player(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    real_put = cache.put

    def put(slug, payload, *a, **k):
        if slug == "element-summary-1":
            raise OSError("disk full")
        return real_put(slug, payload, *a, **k)

    monkeypatch.setattr(cache, "put", put)
    s = PerUrlSession({_es(1): {"id": 1}, _es(2): {"id": 2}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2])

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}         # today: not rescued by a fallback
    assert c.stale is True
    assert sorted(s.calls) == [_es(1), _es(2)]


def test_a_corrupt_fallback_snapshot_fails_only_that_player(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    # A genuinely stale-but-valid snapshot: get_fresh's initial read succeeds
    # and is cleanly rejected for staleness (no exception raised), so the code
    # reaches the network attempt and, on its failure, the except block's OWN
    # newest() call -- which is what this test corrupts, on its second call
    # only, mirroring how test_a_failed_cache_write_fails_only_that_player
    # monkeypatches cache.put for the same reason.
    cache.put("element-summary-1", {"id": 1, "src": "old"}, now=now() - timedelta(days=11))
    real_newest = cache.newest
    calls = {"n": 0}

    def newest(slug, *a, **k):
        calls["n"] += 1
        if slug == "element-summary-1" and calls["n"] > 1:
            raise OSError("second read failed")
        return real_newest(slug, *a, **k)

    monkeypatch.setattr(cache, "newest", newest)
    s = PerUrlSession({_es(2): {"id": 2}}, failing=[_es(1)])
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2], not_before=now() - timedelta(days=2))

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}
    assert s.calls == [_es(1), _es(2)]     # player 1's network attempt DID happen


def test_a_raising_progress_callback_still_stops_the_fetch(tmp_path):
    s = PerUrlSession({_es(p): {"id": p} for p in (1, 2, 3)})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    def progress(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        c.element_summaries([1, 2, 3], progress=progress)


# Concurrent fetch: retries, server backoff, interruption, and duplicate IDs.
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import format_datetime

import requests

import fpl.data.client as client_mod


class ScriptedResponse:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers = payload, status, headers or {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class ScriptedSession:
    """Give each URL its own sequence of responses, safely across threads."""

    def __init__(self, scripts, latency_s: float = 0.0):
        self.scripts = {url: list(steps) for url, steps in scripts.items()}
        self.latency_s = latency_s
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def get(self, url, timeout=None):
        with self.lock:
            self.calls.append(url)
            steps = self.scripts[url]
            step = steps.pop(0) if len(steps) > 1 else steps[0]
        if self.latency_s:
            time.sleep(self.latency_s)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted")
        status, headers = step if isinstance(step, tuple) else (step, {})
        return ScriptedResponse({"url": url}, status, headers)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(client_mod, "BACKOFF_S", (0.0, 0.0, 0.0))


def test_transient_503_is_retried_until_it_succeeds(tmp_path, no_backoff):
    session = ScriptedSession({_es(1): [503, 503, 200]})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)

    assert client.element_summaries([1]) == {1: {"url": _es(1)}}
    assert len(session.calls) == 3
    assert client.fetch_failures == set() and client.stale is False


def test_429_pauses_the_limiter_and_retries(tmp_path, no_backoff, monkeypatch):
    session = ScriptedSession({_es(1): [(429, {"Retry-After": "0"}), 200]})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)
    pauses = []
    real = client._limiter.pause_for
    monkeypatch.setattr(client._limiter, "pause_for",
                        lambda seconds: (pauses.append(seconds), real(seconds)))

    assert 1 in client.element_summaries([1])
    assert len(session.calls) == 2
    assert pauses == [0.0]


def test_connection_errors_are_retried(tmp_path, no_backoff):
    session = ScriptedSession({_es(1): [requests.ConnectionError, 200]})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)

    assert 1 in client.element_summaries([1])
    assert len(session.calls) == 2


def test_404_is_not_retried(tmp_path, no_backoff):
    session = ScriptedSession({_es(1): [404]})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)

    assert client.element_summaries([1]) == {}
    assert len(session.calls) == 1
    assert client.fetch_failures == {1}


def test_persistent_503_gives_up_after_bounded_retries(tmp_path, no_backoff):
    cache = Cache(tmp_path)
    cache.put("element-summary-2", {"src": "old"}, now=now() - timedelta(days=11))
    session = ScriptedSession({_es(1): [503], _es(2): [503]})
    client = FplClient(cache, rate_limit_s=0, session=session)

    assert client.element_summaries([1, 2], not_before=now() - timedelta(days=2)) == {
        2: {"src": "old"}}
    assert client.fetch_failures == {1}
    assert session.calls.count(_es(1)) == 4


def test_an_unexpected_error_is_not_retried(tmp_path, no_backoff):
    session = ScriptedSession({_es(1): [RuntimeError]})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)

    assert client.element_summaries([1]) == {}
    assert len(session.calls) == 1


def test_retry_after_accepts_seconds_and_http_dates():
    parse = FplClient._retry_after
    assert parse(ScriptedResponse({}, 429, {"Retry-After": "7"})) == 7.0
    soon = datetime.now(timezone.utc) + timedelta(seconds=90)
    got = parse(ScriptedResponse({}, 429,
                                 {"Retry-After": format_datetime(soon, usegmt=True)}))
    assert 85 <= got <= 91
    past = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert parse(ScriptedResponse({}, 429,
                                  {"Retry-After": format_datetime(past, usegmt=True)})) == 0.0
    assert parse(ScriptedResponse({}, 429, {"Retry-After": "soon"})) is None
    assert parse(ScriptedResponse({}, 429, {})) is None


def test_a_long_retry_after_stops_asking_and_falls_back(tmp_path, no_backoff):
    cache = Cache(tmp_path)
    for pid in (2, 3, 4, 5):
        cache.put(f"element-summary-{pid}", {"src": "old"},
                  now=now() - timedelta(days=11))
    scripts = {_es(1): [(429, {"Retry-After": "3600"})]}
    scripts.update({_es(pid): [200] for pid in (2, 3, 4, 5)})
    session = ScriptedSession(scripts)
    client = FplClient(cache, rate_limit_s=0, session=session, fetch_workers=1)

    out = client.element_summaries([1, 2, 3, 4, 5],
                                   not_before=now() - timedelta(days=2))
    assert session.calls.count(_es(1)) == 1
    assert len(session.calls) <= 2
    assert set(out) == {2, 3, 4, 5}
    assert all(out[pid] == {"src": "old"} for pid in (3, 4, 5))
    assert client.fetch_failures == {1}
    assert client.stale is True


def test_long_retry_after_stops_queued_worker_before_another_request(tmp_path):
    """A worker must signal stop itself, before the caller inspects its future."""
    session = ScriptedSession({
        _es(1): [(429, {"Retry-After": "3600"})],
        _es(2): [200],
    })
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)
    stop = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client._fetch_json, _es(1), stop)
        queued = pool.submit(client._fetch_json, _es(2), stop)
        with pytest.raises(client_mod.ServerBackoff):
            first.result(timeout=2)
        with pytest.raises(client_mod.FetchCancelled):
            queued.result(timeout=2)

    assert stop.is_set()
    assert session.calls == [_es(1)]


class _Stop(Exception):
    pass


def _stop_at(n):
    def progress(done, total):
        if done == n:
            raise _Stop
    return progress


def test_an_interrupted_refresh_stops_its_workers(tmp_path):
    session = ScriptedSession({_es(pid): [200] for pid in range(1, 41)}, latency_s=0.1)
    cache = Cache(tmp_path)
    client = FplClient(cache, rate_limit_s=0, session=session,
                       fetch_workers=2, fetch_rate_per_s=0)

    started = time.monotonic()
    with pytest.raises(_Stop):
        client.element_summaries(range(1, 41), progress=_stop_at(3))
    elapsed = time.monotonic() - started
    calls_at_return = len(session.calls)
    time.sleep(0.3)

    assert elapsed < 1.0
    assert len(session.calls) == calls_at_return
    assert calls_at_return <= 7
    assert sum(bool(cache.newest(f"element-summary-{pid}")) for pid in range(1, 41)) >= 3


def test_an_interrupt_cuts_a_retry_backoff_short(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "BACKOFF_S", (5.0, 5.0, 5.0))
    scripts = {_es(1): [503, 200]}
    scripts.update({_es(pid): [200] for pid in range(2, 9)})
    session = ScriptedSession(scripts, latency_s=0.02)
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session,
                       fetch_workers=2, fetch_rate_per_s=0)

    started = time.monotonic()
    with pytest.raises(_Stop):
        client.element_summaries(range(1, 9), progress=_stop_at(3))
    assert time.monotonic() - started < 1.0
    assert session.calls.count(_es(1)) == 1


def test_duplicate_ids_are_fetched_once(tmp_path):
    session = PerUrlSession({_es(1): {"id": 1}, _es(2): {"id": 2}})
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=session)
    seen = []

    out = client.element_summaries([1, 2, 1],
                                   progress=lambda done, total: seen.append((done, total)))
    assert list(out) == [1, 2]
    assert sorted(session.calls) == [_es(1), _es(2)]
    assert seen[-1] == (2, 2)


def test_rate_limit_zero_leaves_the_fetch_uncapped(tmp_path):
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=PerUrlSession({}))
    assert client._limiter.rate == 0.0
    assert FplClient(Cache(tmp_path), session=PerUrlSession({}))._limiter.rate == 5.0
    assert FplClient(Cache(tmp_path), rate_limit_s=0, session=PerUrlSession({}),
                     fetch_rate_per_s=2)._limiter.rate == 2.0
