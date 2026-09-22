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


def test_data_complete_after_counts_provisionally_finished_matches():
    """FPL sets finished_provisional at the final whistle and `finished` only
    after its bonus/stat check, which can lag by hours. The history rows appear
    at the whistle, so a cache taken during that window is already out of date
    and must not be trusted — GW3 sat in exactly this state, all ten matches
    played with finished=False on every one of them."""
    fx = [
        {"kickoff_time": "2026-08-29T14:00:00Z", "finished": True,
         "finished_provisional": True},
        {"kickoff_time": "2026-09-06T15:30:00Z", "finished": False,
         "finished_provisional": True},
    ]
    got = data_complete_after(fx)
    assert got == datetime(2026, 9, 6, 15, 30, tzinfo=timezone.utc) + timedelta(hours=3)


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


# --- Played is not the same as checked (2026-09-07) ---

def _fx(event, ko, finished, provisional=True):
    return {"event": event, "kickoff_time": ko, "finished": finished,
            "finished_provisional": provisional}


FINAL_GW3 = [_fx(3, "2026-09-06T13:00:00Z", True), _fx(3, "2026-09-06T15:30:00Z", True)]
WHISTLE_GW3 = [_fx(3, "2026-09-06T13:00:00Z", True),
               _fx(3, "2026-09-06T15:30:00Z", False)]


def test_final_through_reports_the_last_fully_checked_gameweek():
    from fpl.data.cache import final_through
    fixtures = [_fx(2, "2026-08-31T19:00:00Z", True)] + FINAL_GW3
    assert final_through(fixtures) == 3


def test_a_gameweek_at_the_whistle_is_not_yet_checked():
    """`finished_provisional` flips at full time; bonus and stat corrections
    land when `finished` does. GW3 2026/27 sat in that state for hours."""
    from fpl.data.cache import final_through
    assert final_through([_fx(2, "2026-08-31T19:00:00Z", True)] + WHISTLE_GW3) == 2


def test_final_through_is_zero_before_a_ball_is_kicked():
    from fpl.data.cache import final_through
    assert final_through([_fx(1, "2026-08-21T19:00:00Z", False, provisional=False)]) == 0


def test_a_snapshot_records_what_was_checked_when_it_was_taken(tmp_path):
    """No timestamp can express this: two snapshots taken a minute apart sit on
    either side of FPL's data check, and only the snapshot itself knows which."""
    c = Cache(tmp_path)
    c.put("bootstrap-static", {"x": 1}, meta={"final_through": 3})
    assert c.newest_meta("bootstrap-static") == {"final_through": 3}


def test_a_snapshot_taken_before_the_check_is_refused(tmp_path):
    c = Cache(tmp_path)
    c.put("element-summary-1", {"history": []}, meta={"final_through": 2})
    assert c.get_fresh("element-summary-1", ttl_hours=720) is not None
    assert c.get_fresh("element-summary-1", ttl_hours=720, require_final_through=3) is None
    assert c.get_fresh("element-summary-1", ttl_hours=720, require_final_through=2) is not None


def test_a_snapshot_from_before_this_mechanism_is_still_used(tmp_path):
    """Refusing every unmarked snapshot would re-fetch 650 players for data
    that is usually already settled. They are used, and flagged instead."""
    c = Cache(tmp_path)
    c.put("element-summary-1", {"history": []})
    assert c.get_fresh("element-summary-1", ttl_hours=720, require_final_through=3) is not None
    assert c.newest_meta("element-summary-1") == {}


def test_the_marker_does_not_look_like_a_snapshot(tmp_path):
    """`_paths` globs for snapshots; a sidecar ending in .json would be read as
    one and blow up on the timestamp parse."""
    c = Cache(tmp_path)
    c.put("fixtures", [{"id": 1}], meta={"final_through": 3})
    assert c.newest("fixtures")[0] == [{"id": 1}]
    assert c.newest_stamp("fixtures") is not None


def test_pruning_takes_the_marker_with_it(tmp_path):
    c = Cache(tmp_path)
    for h in range(5):
        c.put("bootstrap-static", {"n": h},
              now=datetime(2026, 9, 1, h, tzinfo=timezone.utc), meta={"final_through": h})
    c.prune("bootstrap-static", keep=2)
    assert len(list(tmp_path.glob("bootstrap-static_*.json"))) == 2
    assert len(list(tmp_path.glob("bootstrap-static_*.meta"))) == 2


def test_settled_after_allows_for_fpls_own_check():
    from fpl.data.cache import settled_after, FINAL_CHECK_H
    when = settled_after(FINAL_GW3, 3)
    assert when == datetime(2026, 9, 6, 15, 30, tzinfo=timezone.utc) + timedelta(hours=FINAL_CHECK_H)
    assert settled_after(FINAL_GW3, 4) is None


# --- the slug index ---------------------------------------------------------
import os
from pathlib import Path


def test_put_after_the_index_is_built_is_visible(tmp_path):
    c = Cache(tmp_path)
    assert c.newest("fixtures") is None          # builds the (empty) index
    c.put("fixtures", [1], now=NOW - timedelta(hours=2))
    c.put("fixtures", [2], now=NOW - timedelta(hours=1))
    assert c.newest("fixtures")[0] == [2]


def test_prune_updates_the_index(tmp_path):
    c = Cache(tmp_path)
    for h in range(5):
        c.put("bootstrap-static", {"n": h}, now=NOW - timedelta(hours=h))
    c.prune("bootstrap-static", keep=3)
    assert len(c._paths("bootstrap-static")) == 3
    assert all(p.exists() for p in c._paths("bootstrap-static"))
    c.put("bootstrap-static", {"n": -1}, now=NOW + timedelta(hours=1))
    assert c.newest("bootstrap-static")[0] == {"n": -1}


def test_index_ignores_files_that_are_not_snapshots(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1], now=NOW, meta={"final_through": 3})
    (tmp_path / "fixtures_notes.json").write_text("{}")       # no timestamp
    (tmp_path / "fixtures_20260101T000000Z.txt").write_text("x")
    fresh = Cache(tmp_path)                                    # index built from disk
    assert fresh.newest("fixtures")[0] == [1]
    assert len(fresh._paths("fixtures")) == 1


def test_index_keeps_prefix_sharing_slugs_apart(tmp_path):
    c = Cache(tmp_path)
    c.put("element-summary-1", {"id": 1}, now=NOW)
    c.put("element-summary-10", {"id": 10}, now=NOW + timedelta(hours=1))
    fresh = Cache(tmp_path)
    assert fresh.newest("element-summary-1")[0] == {"id": 1}
    assert fresh.newest("element-summary-10")[0] == {"id": 10}


def _set_dir_mtime(path, mtime_ns):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, mtime_ns))


def test_another_writer_is_noticed(tmp_path):
    reader, writer = Cache(tmp_path), Cache(tmp_path)
    assert reader.newest("fixtures") is None                   # index built
    writer.put("fixtures", [9], now=NOW)
    # The filesystem normally moves the directory mtime on create; pin a
    # distinct value so the test does not depend on timestamp granularity.
    _set_dir_mtime(tmp_path, os.stat(tmp_path).st_mtime_ns + 10**9)
    assert reader.newest("fixtures")[0] == [9]


def test_a_snapshot_deleted_elsewhere_heals_instead_of_crashing(tmp_path):
    reader = Cache(tmp_path)
    reader.put("fixtures", ["old"], now=NOW - timedelta(hours=1))
    reader.put("fixtures", ["new"], now=NOW)
    assert reader.newest("fixtures")[0] == ["new"]             # index current
    known = os.stat(tmp_path).st_mtime_ns
    reader._paths("fixtures")[0].unlink()                      # "another process" prunes it
    # Worst case: the deletion landed inside one timestamp tick, so the
    # directory mtime looks unchanged and only the missing file tells.
    _set_dir_mtime(tmp_path, known)
    assert reader.newest("fixtures")[0] == ["old"]


def test_own_writes_do_not_trigger_rescans(tmp_path, monkeypatch):
    c = Cache(tmp_path)
    c.newest("fixtures")                                       # index built
    scans = {"n": 0}
    real_scandir, real_glob = os.scandir, Path.glob

    # Both, because pathlib's glob does not list through os.scandir.
    def counting_scandir(path=".", *a, **k):
        scans["n"] += 1
        return real_scandir(path, *a, **k)

    def counting_glob(self, pattern, *a, **k):
        scans["n"] += 1
        return real_glob(self, pattern, *a, **k)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    monkeypatch.setattr(Path, "glob", counting_glob)
    for h in range(20):
        c.put("fixtures", [h], now=NOW + timedelta(hours=h), meta={"final_through": 1})
        c.prune("fixtures", keep=3)
        c.newest("fixtures")
    assert scans["n"] == 0
    assert c.newest("fixtures")[0] == [19]


def test_refresh_sees_another_writer(tmp_path):
    reader, writer = Cache(tmp_path), Cache(tmp_path)
    assert reader.newest("fixtures") is None
    writer.put("fixtures", [9], now=NOW)
    reader.refresh()
    assert reader.newest("fixtures")[0] == [9]
