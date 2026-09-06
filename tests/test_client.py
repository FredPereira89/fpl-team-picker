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
