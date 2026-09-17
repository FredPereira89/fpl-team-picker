from datetime import datetime, timezone

from fpl.data.snapshots import (capture, load, available, is_point_in_time,
                                contamination_note)

BOOTSTRAP = {"elements": [{"id": 1, "now_cost": 55, "status": "a"}], "events": []}
FIXTURES = [{"id": 1, "event": 5, "finished": False}]
DEADLINE = "2026-09-11T17:30:00Z"


def _when(day, hour=12):
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def test_a_capture_round_trips(tmp_path):
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(10))
    got = load(tmp_path, 5)
    assert got["bootstrap"]["elements"][0]["now_cost"] == 55
    assert got["fixtures"][0]["event"] == 5
    assert got["meta"]["deadline"] == DEADLINE
    assert available(tmp_path) == [5]


def test_the_first_capture_wins(tmp_path):
    """A later run in the same gameweek has seen more team news. Overwriting
    would replace the deadline's knowledge with hindsight, which is the exact
    failure snapshots exist to prevent."""
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(10))
    later = {"elements": [{"id": 1, "now_cost": 99, "status": "i"}], "events": []}
    assert capture(tmp_path, 5, bootstrap=later, fixtures=FIXTURES,
                   deadline=DEADLINE, captured_at=_when(12)) is None
    assert load(tmp_path, 5)["bootstrap"]["elements"][0]["now_cost"] == 55


def test_a_capture_before_the_deadline_is_point_in_time(tmp_path):
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(10))
    assert is_point_in_time(tmp_path, 5) is True


def test_a_capture_after_the_deadline_is_not(tmp_path):
    """It records what was known once the team sheets were out, which is not
    what the manager was deciding on."""
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(20))
    assert is_point_in_time(tmp_path, 5) is False


def test_a_missing_snapshot_is_not_point_in_time(tmp_path):
    assert is_point_in_time(tmp_path, 5) is False
    assert load(tmp_path, 5) is None
    assert available(tmp_path) == []


def test_a_capture_without_a_deadline_cannot_claim_to_be_point_in_time(tmp_path):
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=None, captured_at=_when(10))
    assert is_point_in_time(tmp_path, 5) is False


def test_the_contamination_note_names_the_gameweeks_it_covers(tmp_path):
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(10))
    note = contamination_note(tmp_path, [4, 5, 6])
    assert note and "GW4" in note and "GW6" in note and "GW5" not in note
    assert "UPPER BOUND" in note


def test_a_fully_snapshotted_replay_has_no_caveat(tmp_path):
    capture(tmp_path, 5, bootstrap=BOOTSTRAP, fixtures=FIXTURES,
            deadline=DEADLINE, captured_at=_when(10))
    assert contamination_note(tmp_path, [5]) is None
