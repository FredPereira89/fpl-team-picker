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


def test_a_corrupt_fallback_snapshot_fails_only_that_player(tmp_path):
    cache = Cache(tmp_path)
    _write_corrupt(cache, "element-summary-1", timedelta(days=11))   # stale AND unreadable
    s = PerUrlSession({_es(2): {"id": 2}}, failing=[_es(1)])
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2], not_before=now() - timedelta(days=2))

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}


def test_a_raising_progress_callback_still_stops_the_fetch(tmp_path):
    s = PerUrlSession({_es(p): {"id": p} for p in (1, 2, 3)})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    def progress(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        c.element_summaries([1, 2, 3], progress=progress)
