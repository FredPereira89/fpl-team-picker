from datetime import datetime, timedelta, timezone
import json
from fpl.data.cache import Cache, is_matchday, data_complete_after

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


def test_get_fresh_rejects_cache_written_before_not_before(tmp_path):
    """A payload can be within its TTL and still predate the data it must carry.

    This is the element-summary bug: a 30-day TTL kept returning a snapshot
    taken before GW2 was played, so the model never saw the current season.
    """
    c = Cache(tmp_path)
    c.put("element-summary-1", {"history": []}, now=NOW - timedelta(hours=2))
    assert c.get_fresh("element-summary-1", ttl_hours=720, now=NOW) is not None
    assert c.get_fresh("element-summary-1", ttl_hours=720, now=NOW,
                       not_before=NOW - timedelta(hours=1)) is None


def test_get_fresh_accepts_cache_written_after_not_before(tmp_path):
    c = Cache(tmp_path)
    c.put("element-summary-1", {"history": [1]}, now=NOW - timedelta(minutes=30))
    assert c.get_fresh("element-summary-1", ttl_hours=720, now=NOW,
                       not_before=NOW - timedelta(hours=1)) == {"history": [1]}


def test_data_complete_after_is_last_finished_kickoff_plus_match():
    fx = [
        {"kickoff_time": "2026-08-15T14:00:00Z", "finished": True},
        {"kickoff_time": "2026-08-16T16:30:00Z", "finished": True},
        {"kickoff_time": "2026-08-22T14:00:00Z", "finished": False},
    ]
    got = data_complete_after(fx)
    assert got == datetime(2026, 8, 16, 16, 30, tzinfo=timezone.utc) + timedelta(hours=3)


def test_data_complete_after_ignores_unfinished_and_null_kickoffs():
    fx = [{"kickoff_time": None, "finished": True},
          {"kickoff_time": "2026-08-22T14:00:00Z", "finished": False}]
    assert data_complete_after(fx) is None


def test_data_complete_after_none_pre_season():
    """No finished fixture means nothing to be stale against — TTL stays in charge."""
    assert data_complete_after([]) is None


def test_slug_with_underscore_works_with_newest(tmp_path):
    """Bug 2: slug containing underscore should work correctly with newest()."""
    c = Cache(tmp_path)
    c.put("element_summary", {"x": 1}, now=NOW)
    payload, ts = c.newest("element_summary")
    assert payload == {"x": 1}
    assert ts.tzinfo is not None
