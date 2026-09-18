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
    clean_sheet = np.array([[True], [True]])   # per-player now; see R10 4th review
    saves = np.zeros((2, 1))
    cards = np.zeros((2, 1))
    dc = np.zeros((2, 1))
    conceded_on = np.zeros((2, 1))

    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc, conceded_on)
    assert bps[0, 0] > 0            # played 60+, clean sheet -- has a real score
    assert bps[1, 0] == -np.inf     # never played


def test_clean_sheet_is_per_player_not_per_side():
    """R10 4th review (Codex): the official rule is no goal conceded WHILE
    ON THE PITCH, not the match's final score -- a player subbed at 60'
    keeps his clean sheet even if his side concedes afterwards. Two
    defenders, same side, same match: one has `conceded_on == 0` (subbed
    before the goal), the other does not (still on when it went in). Only
    the first should get the GKP/DEF clean-sheet BPS bonus, proving
    `clean_sheet` is read per row, not as one flag for the whole side."""
    from fpl.model.bps import score_side_bps

    positions = np.array(["DEF", "DEF"])
    played = np.array([[True], [True]])
    reached_60 = np.array([[True], [True]])
    goals = np.zeros((2, 1))
    assists = np.zeros((2, 1))
    saves = np.zeros((2, 1))
    cards = np.zeros((2, 1))
    dc = np.zeros((2, 1))
    conceded_on = np.array([[0.0], [1.0]])       # side conceded once, after player 0 left
    clean_sheet = np.array([[True], [False]])    # per-player: derived from conceded_on == 0

    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, cards, dc, conceded_on)
    # Player 0: appearance (6) + clean sheet (12) = 18, no conceded penalty.
    assert bps[0, 0] == pytest.approx(18.0)
    # Player 1: appearance (6), no clean sheet, minus one goal conceded (-4).
    assert bps[1, 0] == pytest.approx(2.0)


# --- R10 second re-review: a position-based BONUS_CALIBRATION was tried and
# reverted here (Codex caught it: `bonus90` is derived from the REAL
# historical `bonus` FPL actually awarded -- model.scoring.RATE_SPECS --
# which is ALREADY the outcome of real match-wide BPS competition, so
# projecting it forward is an ordinary linear extrapolation of an
# already-calibrated statistic, not something that needs a further
# "survival" discount. Applied to the real 2026/27 GW1-4 bootstrap, the
# reverted factors would have cut correctly-awarded bonus totalling
# ~6.4-6.5/match down to ~3.0/match -- 46% of the true total). These tests
# pin the REVERTED, uncalibrated behaviour so it cannot quietly return. ---

def test_expected_bonus_for_has_no_position_parameter():
    """`BONUS_CALIBRATION` and the `position` argument it needed are gone --
    asserted directly so a future patch cannot reintroduce the same
    statistically unsound double-correction without at least this test
    demanding a deliberate decision to add the parameter back."""
    import inspect
    import fpl.model.bps as bpsmod

    params = inspect.signature(bpsmod.expected_bonus_for).parameters
    assert "position" not in params
    assert not hasattr(bpsmod, "BONUS_CALIBRATION")


def test_expected_bonus_for_is_a_plain_linear_projection_of_bonus90():
    """No position-dependent discount: doubling `bonus90` must exactly
    double the projection (below the 3.0 cap), since it is a linear
    extrapolation of an already-calibrated historical rate."""
    from fpl.model.bps import expected_bonus_for

    single = expected_bonus_for(0.5, 80.0, att_mult=1.0)
    double = expected_bonus_for(1.0, 80.0, att_mult=1.0)
    assert double == pytest.approx(2 * single)


def test_build_xp_bonus_term_is_uncalibrated_bonus90_linear_projection():
    """`build_xp`'s bonus term for xp_next1 (Mode 1's default squad-
    selection objective) must equal the plain `expected_bonus_for` value --
    no position-based discount reaching this call site. Isolates the bonus
    term exactly: two players identical except for `bonus90` (everything
    else driving xp_next1 -- goals, assists, DC, cards -- is zero, so it
    cannot move), so the difference in xp_next1 must equal the plain
    difference in expected_bonus_for."""
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

    expected_gap = (expected_bonus_for(1.0, 85.0, att_mult=1.0)
                    - expected_bonus_for(0.0, 85.0, att_mult=1.0))
    assert xp_gap == pytest.approx(expected_gap, abs=1e-3)   # build_xp rounds to 4dp


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
    clean_sheet = np.array([[False], [False]])   # per-player now; see R10 4th review
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


def _side_bps(rows):
    """`score_side_bps` for ONE side (`was_home` shared by every row) of one
    fixture, real yellow/red cards weighted at their own official BPS
    values rather than conflated into one count, `defensive_contribution`
    for `dc` (matching production dc90's actual source, NOT a sum of the
    three raw actions -- see R10's 4th review), and PER-PLAYER clean sheet
    from FPL's own `clean_sheets` field (the official rule is no goal
    conceded WHILE ON THE PITCH, not the match's final score, so a player
    subbed before a later concession keeps it -- a single side-wide flag
    denied it to anyone subbed before that point)."""
    import numpy as np
    from fpl.model.bps import score_side_bps, BPS_CARD, BPS_RED_CARD

    positions = np.array([r["position"] for r in rows])
    played = np.array([[float(r["minutes"]) > 0] for r in rows])
    reached_60 = np.array([[float(r["minutes"]) >= 60] for r in rows])
    goals = np.array([[float(r["goals_scored"])] for r in rows])
    assists = np.array([[float(r["assists"])] for r in rows])
    saves = np.array([[float(r["saves"])] for r in rows])
    dc = np.array([[float(r.get("defensive_contribution", 0) or 0)] for r in rows])
    conceded_on = np.array([[float(r["goals_conceded"])] for r in rows])
    clean_sheet = np.array([[float(r.get("clean_sheets", 0) or 0) > 0] for r in rows])

    # score_side_bps takes one blended `cards` count and a single BPS_CARD
    # weight (production cannot split yellow/red from one simulated draw);
    # real data CAN split them, so pass cards=0 here and add each card's
    # own official BPS value directly afterward instead.
    zero_cards = np.zeros((len(rows), 1))
    bps = score_side_bps(positions, played, reached_60, goals, assists,
                         clean_sheet, saves, zero_cards, dc, conceded_on)
    cards_bps = np.array([[float(r["yellow_cards"]) * BPS_CARD
                          + float(r["red_cards"]) * BPS_RED_CARD] for r in rows])
    return bps + cards_bps


def test_bps_approximation_beats_a_reasonable_floor_on_real_fixtures():
    """Reproduces `scripts/validate_bps.py`'s own metrics on the frozen
    sample: at the time this was written, the full 40-fixture live cache
    scored 45.0% exact bonus-recipient match, 0.690 mean Jaccard, 3.66 BPS
    MAE (DC weights cross-validated leave-one-gameweek-out against the
    CORRECT `defensive_contribution` feature -- not the earlier validator's
    CBI+recoveries+tackles sum, a materially different number for GKP/DEF
    that was validating a feature the shipped model never actually sees;
    see `logo_cv` and the `BPS_DC_ACTION` docstring). The exact-match rate
    is inherently volatile at this sample's size (6 fixtures; one
    genuinely wrong bonus recipient swings it by ~17 points of
    percentage), so its floor here is loose -- Jaccard and MAE are the
    more stable signals and are asserted more tightly.

    Each SIDE of a fixture (grouped by `was_home`) is scored separately --
    an earlier version of this test used one clean-sheet flag for the
    whole match, which credited the LOSING side's defenders with a clean
    sheet whenever the first player happened to be on the winning side."""
    import numpy as np
    from fpl.model.bps import award_match_bonus

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
        home = [r for r in grp if r["was_home"]]
        away = [r for r in grp if not r["was_home"]]
        sides = [s for s in (home, away) if s]

        match_rows, match_bps = [], []
        for side in sides:
            match_rows.extend(side)
            match_bps.append(_side_bps(side))
        approx = np.concatenate(match_bps, axis=0)

        real_bps = np.array([float(r["bps"]) for r in match_rows])
        real_bonus = np.array([float(r["bonus"]) for r in match_rows])
        abs_errs.extend(np.abs(approx[:, 0] - real_bps).tolist())

        approx_bonus = award_match_bonus(approx)[:, 0]
        real_recipients = {r["player_id"] for r, b in zip(match_rows, real_bonus) if b > 0}
        approx_recipients = {r["player_id"] for r, b in zip(match_rows, approx_bonus) if b > 0}
        if real_recipients == approx_recipients:
            exact_match += 1
        union = real_recipients | approx_recipients
        jaccards.append(len(real_recipients & approx_recipients) / len(union) if union else 1.0)

    assert n_fixtures >= 5, "the frozen sample should cover several fixtures"
    assert exact_match / n_fixtures >= 0.1
    assert np.mean(jaccards) >= 0.5
    assert np.mean(abs_errs) <= 5.0
