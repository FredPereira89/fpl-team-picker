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


def test_naive_datetime_treated_as_utc_in_is_matchday():
    """Bug 1: naive datetime should be treated as UTC, not local system time."""
    naive_now = datetime(2026, 8, 21, 0, 30)  # intended as UTC, no tzinfo
    fx = [{"kickoff_time": "2026-08-21T22:00:00Z"}]
    # Should be True because it's the same date in UTC
    assert is_matchday(fx, naive_now) is True


def test_naive_datetime_treated_as_utc_in_put_and_get_fresh(tmp_path):
    """Bug 1: naive datetime in put/get_fresh should be treated as UTC, not local time."""
    c = Cache(tmp_path)
    # Use naive datetime intended as UTC
    naive_put_time = datetime(2026, 8, 21, 10, 0)  # no tzinfo, intended as UTC
    c.put("test-data", {"value": 42}, now=naive_put_time)

    # Use naive datetime intended as UTC for get_fresh check
    naive_check_time = datetime(2026, 8, 21, 11, 0)  # 1 hour later
    result = c.get_fresh("test-data", ttl_hours=2, now=naive_check_time)
    assert result == {"value": 42}


def test_slug_with_underscore_works_with_newest(tmp_path):
    """Bug 2: slug containing underscore should work correctly with newest()."""
    c = Cache(tmp_path)
    c.put("element_summary", {"x": 1}, now=NOW)
    payload, ts = c.newest("element_summary")
    assert payload == {"x": 1}
    assert ts.tzinfo is not None
