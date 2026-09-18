import pandas as pd
import pytest
from fpl.model.bps import expected_bonus

RATES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "bonus90": [1.2, 0.1, 0.0],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "e_minutes": [85.0, 85.0, 0.0],
})


def test_high_bps_player_earns_more_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[1] > b.loc[2]


def test_zero_minutes_earns_no_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[3] == 0.0


def test_bonus_scales_with_expected_minutes():
    half = MINUTES.copy()
    half.loc[half.player_id == 1, "e_minutes"] = 42.5
    assert expected_bonus(RATES, half).loc[1] < expected_bonus(RATES, MINUTES).loc[1]


def test_favourable_fixture_multiplier_increases_bonus():
    base = expected_bonus(RATES, MINUTES).loc[1]
    boosted = expected_bonus(RATES, MINUTES, att_mult=1.4).loc[1]
    assert boosted > base


def test_bonus_never_exceeds_three_per_match():
    hot = pd.DataFrame({"player_id": [1], "bonus90": [99.0]})
    mins = pd.DataFrame({"player_id": [1], "e_minutes": [90.0]})
    assert expected_bonus(hot, mins, att_mult=3.0).loc[1] <= 3.0


def test_returns_series_indexed_by_player_id():
    b = expected_bonus(RATES, MINUTES)
    assert b.index.name == "player_id"
    assert set(b.index) == {1, 2, 3}


def test_scalar_bonus_matches_the_frame_version():
    """build_xp needs one player's bonus per fixture. Re-filtering the whole
    rates and minutes frames for every (player, fixture) pair made that an
    O(n^2) scan; the scalar form is what the loop should call."""
    from fpl.model.bps import expected_bonus_for
    frame = expected_bonus(RATES, MINUTES, att_mult=1.2)
    for pid, bonus90, e_min in zip(RATES["player_id"], RATES["bonus90"],
                                   MINUTES["e_minutes"]):
        assert expected_bonus_for(bonus90, e_min, 1.2) == pytest.approx(frame.loc[pid])


def test_scalar_bonus_is_capped_at_three():
    from fpl.model.bps import expected_bonus_for
    assert expected_bonus_for(9.0, 90.0, 2.0) == 3.0


# --- R10: match-wide bonus ranking, FPL's own tie rules ------------------
# Official examples (premierleague.com): a tie for 1st gives the tied
# players 3 each and skips straight to 1 for the next; a tie for 2nd gives
# 3/2/2; a tie for 3rd gives 3/2/1/1.

import numpy as np
from fpl.model.bps import award_match_bonus


def _col(values):
    """One scenario's BPS values as the (n_players, 1) shape award_match_bonus
    expects."""
    return np.array(values, dtype=float)[:, None]


def test_no_ties_awards_three_two_one():
    bonus = award_match_bonus(_col([40.0, 30.0, 20.0, 10.0]))
    assert list(bonus[:, 0]) == [3.0, 2.0, 1.0, 0.0]


def test_a_tie_for_first_gives_both_three_and_skips_to_one():
    bonus = award_match_bonus(_col([40.0, 40.0, 20.0, 10.0]))
    assert list(bonus[:, 0]) == [3.0, 3.0, 1.0, 0.0]


def test_a_tie_for_second_gives_both_two():
    bonus = award_match_bonus(_col([40.0, 20.0, 20.0, 10.0]))
    assert list(bonus[:, 0]) == [3.0, 2.0, 2.0, 0.0]


def test_a_tie_for_third_gives_both_one():
    bonus = award_match_bonus(_col([40.0, 30.0, 20.0, 20.0, 10.0]))
    assert list(bonus[:, 0]) == [3.0, 2.0, 1.0, 1.0, 0.0]


def test_a_three_way_tie_for_first_gives_all_three_and_nothing_lower():
    bonus = award_match_bonus(_col([40.0, 40.0, 40.0, 10.0]))
    assert list(bonus[:, 0]) == [3.0, 3.0, 3.0, 0.0]


def test_non_appearing_players_never_receive_bonus_even_with_few_scorers():
    """Reproduced directly: a `-inf`-marked non-appearing player ties with
    every OTHER non-appearing player, and on `strictly_ahead` alone (how
    many players have a strictly greater score) that tied group can still
    inherit whatever rank position the real scorers left open -- if a match
    has fewer than three real scorers, the non-appearing group would
    otherwise be handed the leftover bonus. No one who did not play may
    ever score, regardless of how few genuine candidates exist."""
    bonus = award_match_bonus(_col([40.0, 30.0, -np.inf, -np.inf, -np.inf]))
    assert list(bonus[:, 0]) == [3.0, 2.0, 0.0, 0.0, 0.0]


def test_vectorised_across_scenarios_independently():
    bps = np.array([[40.0, 10.0], [40.0, 20.0], [10.0, 30.0]])   # 3 players, 2 sims
    bonus = award_match_bonus(bps)
    assert list(bonus[:, 0]) == [3.0, 3.0, 1.0]     # scenario 0: tie for 1st -> 3,3,1
    assert list(bonus[:, 1]) == [1.0, 2.0, 3.0]     # scenario 1: no tie, all 3 score


def test_score_side_bps_excludes_a_player_who_never_played():
    """`_simulate` passes every player on a club's ROSTER, not just those who
    appeared this gameweek -- a non-appearing player scores 0 on every
    component below, which is otherwise indistinguishable from a real 0 and
    would tie for the match lead against other non-appearers."""
    from fpl.model.bps import score_side_bps

    positions = np.array(["DEF", "DEF"])
    played = np.array([[True], [False]])
    reached_60 = np.array([[True], [False]])
    goals = np.zeros((2, 1))
    assists = np.zeros((2, 1))
    clean_sheet = np.array([True])
    saves = np.zeros((2, 1))
    cards = np.zeros((2, 1))
    dc = np.zeros((2, 1))

    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc)
    assert bps[0, 0] > 0            # played 60+, clean sheet -- has a real score
    assert bps[1, 0] == -np.inf     # never played
