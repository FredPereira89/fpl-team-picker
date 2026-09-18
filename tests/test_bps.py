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
    conceded_on = np.zeros((2, 1))

    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc, conceded_on)
    assert bps[0, 0] > 0            # played 60+, clean sheet -- has a real score
    assert bps[1, 0] == -np.inf     # never played


# --- R10 re-review: the analytic bonus90 estimate calibrated toward the
# match-ranked reality (Codex: Mode 1 still used the old, uncorrected
# independent-rate number, which overstates true expected bonus) ---

def test_position_calibration_lowers_the_uncalibrated_estimate():
    """`BONUS_CALIBRATION` factors are all < 1: an independent-rate estimate
    always overstates what match-wide ranking actually delivers (see the
    module docstring for why -- confirmed with a uniform bonus90 test pool
    summing to ~3x the true per-match total)."""
    from fpl.model.bps import expected_bonus_for, BONUS_CALIBRATION

    for position in ("GKP", "DEF", "MID", "FWD"):
        uncalibrated = expected_bonus_for(0.5, 80.0, att_mult=1.0)
        calibrated = expected_bonus_for(0.5, 80.0, att_mult=1.0, position=position)
        assert calibrated < uncalibrated
        assert calibrated == pytest.approx(uncalibrated * BONUS_CALIBRATION[position])


def test_no_position_given_keeps_the_old_uncalibrated_behaviour():
    """A caller that does not know the position (or predates this change)
    gets the OLD number, not a silent, unexplained behaviour change."""
    from fpl.model.bps import expected_bonus_for

    assert expected_bonus_for(0.5, 80.0, att_mult=1.0) == pytest.approx(
        expected_bonus_for(0.5, 80.0, att_mult=1.0, position=None))


def test_build_xp_uses_the_calibrated_bonus_not_the_old_independent_rate():
    """Codex's re-review, finding 1: build_xp (Mode 1's default squad-
    selection objective) called expected_bonus_for without a position,
    so R10's correction never reached the primary decision path -- only
    the rank layer's simulation.

    Isolates the bonus term exactly: two players identical except for
    `bonus90` (everything else driving xp_next1 -- goals, assists, DC,
    cards -- is zero, so it cannot move). The DIFFERENCE in xp_next1 must
    equal the CALIBRATED difference in expected bonus, not the old
    uncalibrated one.
    """
    import pandas as pd
    from fpl.config import Config
    from fpl.model.bps import expected_bonus_for
    from fpl.model.xp import build_xp

    players = pd.DataFrame({"player_id": [1, 2], "web_name": ["P1", "P2"],
                            "team": ["A", "A"], "team_id": [1, 1],
                            "position": ["MID", "MID"], "price": [8.0, 8.0],
                            "selected_by_percent": [10.0, 10.0]})
    rates = pd.DataFrame({"player_id": [1, 2], "xg90": [0.0, 0.0], "xa90": [0.0, 0.0],
                          "bonus90": [0.0, 1.0], "dc90": [0.0, 0.0],
                          "saves90": [0.0, 0.0], "cards90": [0.0, 0.0]})
    minutes = pd.DataFrame({"player_id": [1, 2], "p_start": [0.9, 0.9],
                            "p_play": [0.92, 0.92], "p_60": [0.8, 0.8],
                            "m_start": [90.0, 90.0], "e_minutes": [85.0, 85.0],
                            "confidence": ["high", "high"], "flags": [[], []]})
    tfx = pd.DataFrame([{"team_id": 1, "event": 1, "fixture_id": 1, "opponent_id": 2,
                         "is_home": True, "xgc": 1.0, "p_cs": 0.3, "att_mult": 1.0,
                         "opp_threat": 1.0}])
    cfg = Config(horizon_gw=1)

    xp = build_xp(players, rates, minutes, tfx, cfg, from_event=1).set_index("player_id")
    xp_gap = float(xp.loc[2, "xp_next1"]) - float(xp.loc[1, "xp_next1"])

    calibrated_gap = (expected_bonus_for(1.0, 85.0, att_mult=1.0, position="MID")
                      - expected_bonus_for(0.0, 85.0, att_mult=1.0, position="MID"))
    uncalibrated_gap = (expected_bonus_for(1.0, 85.0, att_mult=1.0)
                        - expected_bonus_for(0.0, 85.0, att_mult=1.0))
    assert calibrated_gap < uncalibrated_gap   # sanity: calibration bites here
    assert xp_gap == pytest.approx(calibrated_gap, abs=1e-3)   # build_xp rounds to 4dp


def test_score_side_bps_penalises_goals_conceded_for_gkp_and_def():
    """2026/27 official rule: -4 BPS per goal conceded for GKP/DEF (not per
    two, unlike the separate FPL POINTS penalty in model.xp) -- Codex's
    re-review found this omitted even though `conceded_on` was already
    being computed for that points penalty."""
    from fpl.model.bps import score_side_bps

    positions = np.array(["DEF", "MID"])
    played = np.array([[True], [True]])
    reached_60 = np.array([[True], [True]])
    goals = np.zeros((2, 1))
    assists = np.zeros((2, 1))
    clean_sheet = np.array([False])
    saves = np.zeros((2, 1))
    cards = np.zeros((2, 1))
    dc = np.zeros((2, 1))
    conceded_on = np.array([[2.0], [2.0]])   # both were on for 2 goals against

    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc, conceded_on)
    # DEF: appearance (6) - 2 goals conceded * 4 = 6 - 8 = -2
    assert bps[0, 0] == pytest.approx(-2.0)
    # MID: not a conceded-penalty position, so only the appearance points
    assert bps[1, 0] == pytest.approx(6.0)


# --- Predictive validation against REAL fixtures (Codex's re-review,
# finding 3): a small, git-tracked sample of real GW1-4 events, frozen from
# `scripts/validate_bps.py`'s live-cache extraction (data/cache is
# gitignored, so a live run cannot be relied on in a fresh checkout or CI).
# Re-run `scripts/validate_bps.py` directly against the live cache for the
# full, current picture; this pins a lower bound so a future coefficient
# change cannot silently make the approximation WORSE without a test
# noticing, without demanding exact reproduction of one small sample. ---

def _real_bps_sample():
    import json
    from pathlib import Path
    path = Path(__file__).parent / "data" / "real_bps_sample.json"
    return json.load(open(path, encoding="utf-8"))


def test_bps_approximation_beats_a_reasonable_floor_on_real_fixtures():
    """Reproduces `scripts/validate_bps.py`'s own metrics on the frozen
    sample: at the time this was written (post-calibration), the full live
    cache scored 42.5% exact bonus-recipient match, 0.675 mean Jaccard,
    3.28 BPS MAE -- this asserts generous floors on the same metrics for
    the smaller frozen sample, loose enough not to be a flaky exact
    reproduction, tight enough to catch a real regression (e.g. reverting
    the conceded-goal penalty, or a sign error in a coefficient)."""
    import numpy as np
    from fpl.model.bps import score_side_bps, award_match_bonus

    rows = _real_bps_sample()
    by_fixture: dict = {}
    for r in rows:
        by_fixture.setdefault(r["fixture"], []).append(r)

    exact_match = 0
    jaccards = []
    abs_errs = []
    n_fixtures = 0
    for fixture_id, grp in by_fixture.items():
        n_fixtures += 1
        positions = np.array([r["position"] for r in grp])
        played = np.array([[float(r["minutes"]) > 0] for r in grp])
        reached_60 = np.array([[float(r["minutes"]) >= 60] for r in grp])
        goals = np.array([[float(r["goals_scored"])] for r in grp])
        assists = np.array([[float(r["assists"])] for r in grp])
        saves = np.array([[float(r["saves"])] for r in grp])
        cards = np.array([[float(r["yellow_cards"]) + float(r["red_cards"])] for r in grp])
        dc = np.array([[float(r["clearances_blocks_interceptions"]) + float(r["recoveries"])
                       + float(r["tackles"])] for r in grp])
        conceded_on = np.array([[float(r["goals_conceded"])] for r in grp])
        # Team clean sheet: opponent's score, from whichever side this row's
        # player was on.
        opp_scores = [float(r["team_a_score"]) if r["was_home"] == "True" or r["was_home"] is True
                     else float(r["team_h_score"]) for r in grp]
        clean_sheet = np.array([opp_scores[0] == 0.0])   # same match, same value

        approx = score_side_bps(positions, played, reached_60, goals, assists,
                                clean_sheet, saves, cards, dc, conceded_on)
        real_bps = np.array([float(r["bps"]) for r in grp])
        real_bonus = np.array([float(r["bonus"]) for r in grp])
        abs_errs.extend(np.abs(approx[:, 0] - real_bps).tolist())

        approx_bonus = award_match_bonus(approx)[:, 0]
        real_recipients = {r["player_id"] for r, b in zip(grp, real_bonus) if b > 0}
        approx_recipients = {r["player_id"] for r, b in zip(grp, approx_bonus) if b > 0}
        if real_recipients == approx_recipients:
            exact_match += 1
        union = real_recipients | approx_recipients
        jaccards.append(len(real_recipients & approx_recipients) / len(union) if union else 1.0)

    assert n_fixtures >= 5, "the frozen sample should cover several fixtures"
    assert exact_match / n_fixtures >= 0.25
    assert np.mean(jaccards) >= 0.5
    assert np.mean(abs_errs) <= 5.0
