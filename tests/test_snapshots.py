from datetime import datetime, timezone

from fpl.data.snapshots import (capture, load, select, versions, available,
                                is_point_in_time, contamination_note)

BOOTSTRAP = {"elements": [{"id": 1, "now_cost": 55, "status": "a"}], "events": []}
FIXTURES = [{"id": 1, "event": 5, "finished": False}]
DEADLINE = "2026-09-11T17:30:00Z"


def _when(day, hour=12):
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def _cap(root, cost=55, day=10, deadline=DEADLINE):
    boot = {"elements": [{"id": 1, "now_cost": cost, "status": "a"}], "events": []}
    return capture(root, 5, bootstrap=boot, fixtures=FIXTURES,
                   deadline=deadline, captured_at=_when(day))


def test_a_capture_round_trips(tmp_path):
    v = _cap(tmp_path)
    got = load(tmp_path, 5)
    assert got["bootstrap"]["elements"][0]["now_cost"] == 55
    assert got["fixtures"][0]["event"] == 5
    assert got["meta"]["deadline"] == DEADLINE
    assert got["meta"]["version"] == v
    assert available(tmp_path) == [5]


def test_every_capture_is_kept_as_its_own_version(tmp_path):
    """A later PRE-deadline run carries legitimate team news -- it is the run the
    manager acted on. Keeping only the first capture would replay inputs the
    actioned forecast never saw."""
    _cap(tmp_path, cost=55, day=9)
    _cap(tmp_path, cost=57, day=10)
    assert len(versions(tmp_path, 5)) == 2
    assert load(tmp_path, 5)["bootstrap"]["elements"][0]["now_cost"] == 57


def test_selection_prefers_the_newest_capture_before_the_deadline(tmp_path):
    _cap(tmp_path, cost=55, day=9)
    _cap(tmp_path, cost=57, day=10)
    _cap(tmp_path, cost=99, day=12)        # after the deadline: hindsight
    assert load(tmp_path, 5, deadline=DEADLINE)["bootstrap"]["elements"][0]["now_cost"] == 57


def test_a_named_version_wins(tmp_path):
    first = _cap(tmp_path, cost=55, day=9)
    _cap(tmp_path, cost=57, day=10)
    assert load(tmp_path, 5, version=first)["bootstrap"]["elements"][0]["now_cost"] == 55
    assert select(tmp_path, 5, version="nope") is None


def test_a_capture_before_the_deadline_is_point_in_time(tmp_path):
    _cap(tmp_path, day=10)
    assert is_point_in_time(tmp_path, 5) is True


def test_only_a_post_deadline_capture_is_not_point_in_time(tmp_path):
    """It records what was known once the team sheets were out, which is not
    what the manager was deciding on."""
    _cap(tmp_path, day=20)
    assert is_point_in_time(tmp_path, 5) is False


def test_a_missing_snapshot_is_not_point_in_time(tmp_path):
    assert is_point_in_time(tmp_path, 5) is False
    assert load(tmp_path, 5) is None
    assert available(tmp_path) == []


def test_a_capture_without_a_deadline_cannot_claim_to_be_point_in_time(tmp_path):
    _cap(tmp_path, day=10, deadline=None)
    assert is_point_in_time(tmp_path, 5) is False


def test_the_contamination_note_names_the_gameweeks_it_covers(tmp_path):
    _cap(tmp_path, day=10)
    note = contamination_note(tmp_path, [4, 5, 6])
    assert note and "GW4" in note and "GW6" in note and "GW5" not in note
    assert "UPPER BOUND" in note


def test_a_fully_snapshotted_replay_has_no_caveat(tmp_path):
    _cap(tmp_path, day=10)
    assert contamination_note(tmp_path, [5]) is None


def test_the_note_honours_a_pinned_post_deadline_version(tmp_path):
    _cap(tmp_path, day=10)
    late = _cap(tmp_path, day=20)
    assert contamination_note(tmp_path, [5], versions_by_gw={5: late}) is not None


def test_a_capture_keeps_the_overrides_and_config_the_run_used(tmp_path):
    """Production minutes use the manual news overrides; a replay that recomputed
    them without the overrides reconstructed a decision the live run never made."""
    news = {42: {"p_start_override": 0.9, "note": "starts", "source": "presser"}}
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES, deadline=DEADLINE,
            captured_at=_when(10), news=news, config={"horizon_gw": 5})
    got = load(tmp_path, 5)
    assert got["news"] == news
    assert got["config"]["horizon_gw"] == 5
