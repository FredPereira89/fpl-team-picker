import pytest
from fpl.data.cache import Cache
from fpl.data.client import FplClient, BASE


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
