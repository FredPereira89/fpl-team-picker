import math
import pandas as pd
from fpl.config import Config
import pytest
from fpl.model.scoring import (per90_rates, blended_rates, apply_set_piece_roles,
                                ew_mean, form_weight, blend_form)

CFG = Config(shrinkage_minutes=900, form_half_life_gw=3, form_max_weight=0.6)

# 8 filler MID players added beyond the brief's original 3-player fixture.
# Root cause: with only 2 MID players (player 1 at 3000 mins/raw 0.45, player 2
# at 90 mins/raw 1.0), the position mean self-includes player 2's noisy small-
# sample outlier, pulling player 1's shrunk rate to ~0.51 -- outside the
# 0.35-0.46 band the first test asserts. Real leagues have ~50+ players per
# position so one outlier barely moves the mean; this fixture didn't reflect
# that. Adding 8 realistic, larger-sample MID fillers (raw rate 0.25, i.e.
# unremarkable non-scoring midfielders) brings the pool mean down to a
# realistic level without changing per90_rates' code at all.
PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "position": ["MID", "MID", "GKP"] + ["MID"] * 8,
    "minutes": [3000, 90, 3000] + [3000] * 8,
    "expected_goals": [15.0, 1.0, 0.0] + [8.333333333333334] * 8,
    "expected_assists": [10.0, 0.5, 0.0] + [0.0] * 8,
    "bonus": [30, 1, 12] + [0] * 8,
    "saves": [0, 0, 100] + [0] * 8,
    "yellow_cards": [4, 0, 1] + [0] * 8,
    "red_cards": [0, 0, 0] + [0] * 8,
    "defensive_contribution": [380, 12, 0] + [0] * 8,
})


def test_high_minutes_player_keeps_own_rate():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    # 15 xG over 3000 mins = 0.45/90; shrinkage pulls it only slightly
    assert 0.35 < r.loc[1, "xg90"] < 0.46


def test_tiny_sample_is_shrunk_toward_position_mean():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    # player 2 raw rate is 1.0 xG/90 but only 90 minutes -> heavily shrunk down
    assert r.loc[2, "xg90"] < 0.5


def test_shrinkage_is_monotonic_in_minutes():
    low = PLAYERS.copy()
    low.loc[low.player_id == 2, "minutes"] = 90
    high = PLAYERS.copy()
    high.loc[high.player_id == 2, "minutes"] = 3000
    high.loc[high.player_id == 2, "expected_goals"] = 33.3  # same 1.0/90 raw rate
    r_low = per90_rates(low, CFG).set_index("player_id").loc[2, "xg90"]
    r_high = per90_rates(high, CFG).set_index("player_id").loc[2, "xg90"]
    assert r_high > r_low


def test_saves_only_meaningful_for_keeper():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    assert r.loc[3, "saves90"] > 1.0
    assert r.loc[1, "saves90"] == 0.0


def test_all_rates_non_negative():
    r = per90_rates(PLAYERS, CFG)
    for col in ["xg90", "xa90", "bonus90", "dc90", "saves90", "cards90"]:
        assert (r[col] >= 0).all()


def test_ew_mean_weights_recent_more():
    assert ew_mean([0, 0, 10], half_life=3) > ew_mean([10, 0, 0], half_life=3)


def test_ew_mean_of_constant_series_is_that_constant():
    assert math.isclose(ew_mean([5, 5, 5, 5], half_life=3), 5.0, rel_tol=1e-9)


def test_ew_mean_empty_returns_zero():
    assert ew_mean([], half_life=3) == 0.0


def test_form_weight_is_zero_at_gameweek_one():
    assert form_weight(0, CFG) == 0.0


def test_form_weight_ramps_then_caps():
    assert form_weight(3, CFG) == 0.5 * CFG.form_max_weight
    assert form_weight(6, CFG) == CFG.form_max_weight
    assert form_weight(20, CFG) == CFG.form_max_weight


def test_blend_form_returns_pure_baseline_at_gw1():
    assert blend_form(baseline=5.0, form_value=99.0, gws_played=0, cfg=CFG) == 5.0


def test_blend_form_moves_toward_form_as_season_progresses():
    out = blend_form(baseline=5.0, form_value=10.0, gws_played=6, cfg=CFG)
    assert math.isclose(out, 5.0 * 0.4 + 10.0 * 0.6, rel_tol=1e-9)


# --- P3: season-to-date form (2026-08-27 audit) ---
# Until now blend_form and form_weight were tested but called by nothing:
# the model ran on last season alone and could not see the current one.

CURRENT = pd.DataFrame({
    "player_id": [1, 2],
    "gws_played": [6, 6],
    "minutes": [540.0, 540.0],
    "starts": [6, 6],
    "expected_goals": [6.0, 0.0],    # 1.0 xG per 90 -- far above his baseline
    "expected_assists": [0.0, 0.0],
    "bonus": [0, 0],
    "saves": [0, 0],
    "yellow_cards": [0, 0],
    "red_cards": [0, 0],
    "defensive_contribution": [0, 0],
})


def test_form_pulls_a_rate_toward_this_seasons_output():
    base = per90_rates(PLAYERS, CFG).set_index("player_id")
    blended = blended_rates(PLAYERS, CURRENT, CFG).set_index("player_id")
    assert blended.loc[1, "xg90"] > base.loc[1, "xg90"]
    assert blended.loc[1, "xg90"] < 1.0  # but never all the way -- 6 GWs is not a season


def test_form_leaves_players_with_no_current_data_untouched():
    base = per90_rates(PLAYERS, CFG).set_index("player_id")
    blended = blended_rates(PLAYERS, CURRENT, CFG).set_index("player_id")
    assert blended.loc[3, "xg90"] == pytest.approx(base.loc[3, "xg90"])


def test_before_the_season_starts_the_baseline_is_untouched():
    base = per90_rates(PLAYERS, CFG).set_index("player_id")
    empty = CURRENT.iloc[0:0]
    blended = blended_rates(PLAYERS, empty, CFG).set_index("player_id")
    for col in ["xg90", "xa90", "bonus90", "dc90", "saves90", "cards90"]:
        assert blended[col].equals(base[col])


def test_a_new_signing_is_rated_on_this_season_not_the_positional_mean():
    """A player with no prior Premier League row has every baseline rate shrunk
    to the positional mean. Season-to-date output is the only real evidence
    about him, and until now the model could not use it."""
    newcomer = pd.DataFrame([{
        "player_id": 99, "position": "MID", "minutes": 0,
        "expected_goals": 0.0, "expected_assists": 0.0, "bonus": 0, "saves": 0,
        "yellow_cards": 0, "red_cards": 0, "defensive_contribution": 0,
    }])
    players = pd.concat([PLAYERS, newcomer], ignore_index=True)
    current = pd.DataFrame([{
        "player_id": 99, "gws_played": 6, "minutes": 540.0, "starts": 6,
        "expected_goals": 6.0, "expected_assists": 0.0, "bonus": 0, "saves": 0,
        "yellow_cards": 0, "red_cards": 0, "defensive_contribution": 0,
    }])
    base = per90_rates(players, CFG).set_index("player_id")
    blended = blended_rates(players, current, CFG).set_index("player_id")
    assert blended.loc[99, "xg90"] > base.loc[99, "xg90"]


# --- P4: set-piece roles (2026-08-27 audit) ---

TAKERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "position": ["MID"] * 4,
    "minutes": [3000, 0, 3000, 3000],
    "penalties_order": [1, 1, 2, None],
    "corners_and_indirect_freekicks_order": [None, None, None, 1],
    "direct_freekicks_order": [None, None, None, None],
})
FLAT_RATES = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "xg90": [0.3, 0.3, 0.3, 0.3],
    "xa90": [0.2, 0.2, 0.2, 0.2],
    "bonus90": [0.0] * 4, "dc90": [0.0] * 4,
    "saves90": [0.0] * 4, "cards90": [0.0] * 4,
})


def test_designated_penalty_taker_is_credited_with_expected_goals():
    out = apply_set_piece_roles(FLAT_RATES, TAKERS, CFG).set_index("player_id")
    assert out.loc[1, "xg90"] > 0.3


def test_a_player_off_the_list_is_credited_with_nothing():
    out = apply_set_piece_roles(FLAT_RATES, TAKERS, CFG).set_index("player_id")
    assert out.loc[4, "xg90"] == pytest.approx(0.3)


def test_corner_taker_is_credited_with_assists_not_goals():
    out = apply_set_piece_roles(FLAT_RATES, TAKERS, CFG).set_index("player_id")
    assert out.loc[4, "xa90"] > 0.2


def test_an_established_taker_is_credited_less_than_a_newcomer():
    """A full season of measured xG already contains the penalties he took, so
    crediting him again would count them twice. The premium fills in only the
    part of an estimate that is prior rather than evidence."""
    out = apply_set_piece_roles(FLAT_RATES, TAKERS, CFG).set_index("player_id")
    assert out.loc[2, "xg90"] > out.loc[1, "xg90"]


def test_the_backup_taker_is_credited_less_than_the_first_choice():
    out = apply_set_piece_roles(FLAT_RATES, TAKERS, CFG).set_index("player_id")
    assert out.loc[1, "xg90"] > out.loc[3, "xg90"] > 0.3


def test_missing_set_piece_columns_are_treated_as_no_duty():
    """Older cached snapshots predate these fields."""
    bare = TAKERS.drop(columns=["penalties_order", "corners_and_indirect_freekicks_order",
                                "direct_freekicks_order"])
    out = apply_set_piece_roles(FLAT_RATES, bare, CFG).set_index("player_id")
    assert out.loc[1, "xg90"] == pytest.approx(0.3)
