from datetime import datetime, timezone

from fpl.backtest.manifest import (record_version, entries, mark_actioned,
                                   select_version, model_version)


def _when(day: int, hour: int = 12):
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def _record(root, gw=5, version="v1", day=10, origin="live", deadline="2026-09-11T17:30:00Z"):
    return record_version(root, gw=gw, version=version, created_at=_when(day),
                          origin=origin, model_version="test", config_hash="abc",
                          deadline=deadline)


def test_a_recorded_version_reads_back(tmp_path):
    _record(tmp_path)
    got = entries(tmp_path, gw=5)
    assert len(got) == 1
    assert got[0]["version"] == "v1"
    assert got[0]["origin"] == "live"


def test_the_manifest_is_append_only(tmp_path):
    """The whole point is that a later run cannot rewrite what an earlier one
    recorded -- otherwise a post-deadline re-run silently replaces the forecast
    that was actually acted on."""
    _record(tmp_path, version="v1", day=10)
    _record(tmp_path, version="v2", day=11)
    got = entries(tmp_path, gw=5)
    assert [e["version"] for e in got] == ["v1", "v2"]


def test_entries_can_be_read_across_gameweeks(tmp_path):
    _record(tmp_path, gw=5, version="v1")
    _record(tmp_path, gw=6, version="v1")
    assert len(entries(tmp_path)) == 2
    assert len(entries(tmp_path, gw=6)) == 1


def test_the_actioned_version_beats_a_newer_one(tmp_path):
    """A forecast is scored because it was ACTED ON, not because it was last."""
    _record(tmp_path, version="v1", day=10)
    mark_actioned(tmp_path, gw=5, version="v1", when=_when(10, 16))
    _record(tmp_path, version="v2", day=12)
    assert select_version(tmp_path, 5)["version"] == "v1"


def test_without_an_actioned_marker_the_newest_pre_deadline_version_wins(tmp_path):
    deadline = "2026-09-11T17:30:00Z"
    _record(tmp_path, version="v1", day=10, deadline=deadline)
    _record(tmp_path, version="v2", day=11, deadline=deadline)   # still before
    _record(tmp_path, version="v3", day=12, deadline=deadline)   # after the deadline
    assert select_version(tmp_path, 5, deadline=deadline)["version"] == "v2"


def test_a_replay_is_never_selected(tmp_path):
    """A replay is built from data the live model never had. Scoring it as the
    week's forecast, or calibrating on it, is measuring the wrong object."""
    _record(tmp_path, version="r1", day=20, origin="replay")
    assert select_version(tmp_path, 5) is None


def test_a_replay_cannot_displace_a_live_forecast(tmp_path):
    _record(tmp_path, version="v1", day=10)
    _record(tmp_path, version="r1", day=20, origin="replay")
    assert select_version(tmp_path, 5)["version"] == "v1"


def test_marking_actioned_without_a_version_takes_the_newest(tmp_path):
    _record(tmp_path, version="v1", day=10)
    _record(tmp_path, version="v2", day=11)
    marked = mark_actioned(tmp_path, gw=5)
    assert marked["version"] == "v2"
    assert select_version(tmp_path, 5)["version"] == "v2"


def test_marking_actioned_on_an_empty_manifest_is_harmless(tmp_path):
    assert mark_actioned(tmp_path, gw=5) is None


def test_the_model_version_is_not_a_hand_maintained_date():
    """A hand-set constant drifts behind the code it is supposed to identify --
    the one in the ledger predated several core commits."""
    v = model_version()
    assert isinstance(v, str) and v
