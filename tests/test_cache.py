from datetime import datetime, timedelta, timezone
import json
from fpl.data.cache import Cache, is_matchday

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_put_then_newest_roundtrips(tmp_path):
    c = Cache(tmp_path)
    c.put("bootstrap-static", {"a": 1})
    payload, ts = c.newest("bootstrap-static")
    assert payload == {"a": 1}
    assert ts.tzinfo is not None


def test_get_fresh_returns_payload_within_ttl(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1, 2], now=NOW - timedelta(hours=2))
    assert c.get_fresh("fixtures", ttl_hours=6, now=NOW) == [1, 2]


def test_get_fresh_returns_none_when_stale(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1, 2], now=NOW - timedelta(hours=8))
    assert c.get_fresh("fixtures", ttl_hours=6, now=NOW) is None


def test_get_fresh_returns_none_when_absent(tmp_path):
    assert Cache(tmp_path).get_fresh("nope", ttl_hours=6, now=NOW) is None


def test_prune_keeps_newest_three(tmp_path):
    c = Cache(tmp_path)
    for h in range(6):
        c.put("bootstrap-static", {"n": h}, now=NOW - timedelta(hours=h))
    removed = c.prune("bootstrap-static", keep=3)
    assert removed == 3
    assert len(list(tmp_path.glob("bootstrap-static_*.json"))) == 3
    # newest survives
    assert c.newest("bootstrap-static")[0] == {"n": 0}


def test_prune_is_slug_scoped(tmp_path):
    c = Cache(tmp_path)
    for h in range(4):
        c.put("fixtures", {"n": h}, now=NOW - timedelta(hours=h))
    c.put("bootstrap-static", {"keep": True}, now=NOW)
    c.prune("fixtures", keep=3)
    assert c.newest("bootstrap-static")[0] == {"keep": True}


def test_is_matchday_true_when_fixture_kicks_off_today():
    fx = [{"kickoff_time": "2026-08-21T19:00:00Z"}]
    assert is_matchday(fx, NOW) is True


def test_is_matchday_false_on_empty_day():
    fx = [{"kickoff_time": "2026-08-22T19:00:00Z"}]
    assert is_matchday(fx, NOW) is False


def test_is_matchday_ignores_null_kickoffs():
    assert is_matchday([{"kickoff_time": None}], NOW) is False
