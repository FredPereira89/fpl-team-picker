"""B16: a partial element-summary fetch must not read as a squad of newcomers.

Individual fetch failures were silently swallowed. The missing player's prior
rows were then zeroed and routed to the same price prior as a genuine new
signing, so a partial API outage could erase an established player's history
while the optimizer still returned a confident, legal team -- and the only
signal was a global `stale` flag that says nothing about who is affected.
"""
import pandas as pd
import pytest

from fpl.data.client import DataCoverageError
from fpl.data.normalize import history_status
from fpl.pipeline import coverage_gate


PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "web_name": ["Established", "Newcomer", "Outage", "Fringe"],
    "position": ["MID", "FWD", "DEF", "GKP"],
    "price": [9.0, 7.0, 6.0, 4.5],
    "minutes": [3000, 0, 2800, 0],
})

SUMMARIES = {
    1: {"history_past": [{"season_name": "2025/26", "minutes": 3000}], "history": []},
    # Genuinely new: no Premier League season on record at all.
    2: {"history_past": [], "history": []},
    # Has been here before and simply did not play last season -- a different
    # prior from a newcomer, and the model should not price him off his fee.
    4: {"history_past": [{"season_name": "2023/24", "minutes": 400}], "history": []},
}


def test_a_failed_fetch_is_reported_apart_from_a_newcomer():
    status = history_status(PLAYERS, SUMMARIES, fetch_failed={3}).set_index("player_id")
    assert status.loc[1, "history_status"] == "established"
    assert status.loc[2, "history_status"] == "new_to_league"
    assert status.loc[3, "history_status"] == "fetch_failed"
    assert status.loc[4, "history_status"] == "no_history"


def test_a_failed_fetch_for_an_owned_player_stops_the_run():
    """One corrupted premium estimate can trigger a bad sale or a hit. There is
    no safe way to price a player whose history this run could not read."""
    reasons = coverage_gate(PLAYERS, SUMMARIES, fetch_failed={3}, owned_ids=[1, 3])
    assert reasons
    assert any("Outage" in r or "3" in r for r in reasons)


def test_a_failed_fetch_for_an_unowned_player_is_tolerated_above_the_floor():
    reasons = coverage_gate(PLAYERS, SUMMARIES, fetch_failed={3}, owned_ids=[1, 2],
                            min_coverage=0.5)
    assert reasons == []


def test_coverage_below_the_floor_stops_the_run():
    reasons = coverage_gate(PLAYERS, SUMMARIES, fetch_failed={3}, owned_ids=[1, 2],
                            min_coverage=0.99)
    assert reasons
    assert any("coverage" in r.lower() for r in reasons)


def test_a_clean_run_reports_nothing():
    assert coverage_gate(PLAYERS, {**SUMMARIES, 3: {"history_past": [], "history": []}},
                         fetch_failed=set(), owned_ids=[1, 2]) == []


def test_the_client_records_which_players_failed():
    """`stale` is a single global boolean; it cannot say who is affected."""
    from fpl.data.client import FplClient

    class Flaky(FplClient):
        def element_summary(self, player_id, **kw):
            if int(player_id) == 3:
                raise RuntimeError("503")
            return {"history_past": [], "history": []}

    client = Flaky.__new__(Flaky)
    client.stale = False
    client.fetch_failures = set()
    got = FplClient.element_summaries(client, [1, 2, 3])
    assert set(got) == {1, 2}
    assert client.fetch_failures == {3}


def test_the_coverage_error_is_raisable():
    with pytest.raises(DataCoverageError):
        raise DataCoverageError("test")
