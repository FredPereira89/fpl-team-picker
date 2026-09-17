import pandas as pd
from fpl.backtest.gw_level import evaluate_predictions, captaincy_hit_rate, trust_gate

POS = pd.Series(["MID", "MID", "DEF", "DEF", "FWD", "FWD"], index=range(6))
ACTUAL = pd.Series([10, 2, 8, 1, 12, 3], index=range(6))
GOOD = pd.Series([9, 3, 7, 2, 11, 4], index=range(6))
BAD = pd.Series([2, 10, 1, 8, 3, 12], index=range(6))


def test_good_predictions_have_low_error():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert out["mae"] < 1.5
    assert out["n"] == 6


def test_good_predictions_rank_positively():
    assert evaluate_predictions(GOOD, ACTUAL, POS)["spearman_overall"] > 0.8


def test_inverted_predictions_rank_negatively():
    assert evaluate_predictions(BAD, ACTUAL, POS)["spearman_overall"] < 0


def test_spearman_reported_per_position():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert set(out["spearman_by_position"]) == {"MID", "DEF", "FWD"}


def test_top20_overlap_is_a_fraction():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert 0.0 <= out["top20_overlap"] <= 1.0


def test_captaincy_hit_rate_perfect_when_top_pick_is_top_scorer():
    pred = {1: pd.Series([5, 9], index=[10, 11]), 2: pd.Series([7, 2], index=[10, 11])}
    actual = {1: pd.Series([4, 12], index=[10, 11]), 2: pd.Series([9, 1], index=[10, 11])}
    assert captaincy_hit_rate(pred, actual) == 1.0


def test_captaincy_hit_rate_zero_when_always_wrong():
    pred = {1: pd.Series([9, 5], index=[10, 11])}
    actual = {1: pd.Series([1, 12], index=[10, 11])}
    assert captaincy_hit_rate(pred, actual) == 0.0


def test_trust_gate_passes_when_model_beats_both_baselines():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.5, "FWD": 0.55}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.2, "FWD": 0.25}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}
    out = trust_gate(model, naive, fpl, full_pipeline=True)
    assert out["trusted"] is True
    assert out["failures"] == []


def test_trust_gate_fails_and_names_the_position():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.1, "FWD": 0.55}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.4, "FWD": 0.25}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}
    out = trust_gate(model, naive, fpl, full_pipeline=True)
    assert out["trusted"] is False
    assert any("DEF" in f for f in out["failures"])
    assert "DEF" in out["summary"]


def test_trust_gate_fails_when_baseline_missing_for_a_position():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.5, "FWD": 0.55, "GKP": 0.05}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.2, "FWD": 0.25}}  # no GKP
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}    # no GKP
    out = trust_gate(model, naive, fpl, full_pipeline=True)
    assert out["trusted"] is False
    assert any("GKP" in f for f in out["failures"])


def test_trust_gate_fails_when_model_missing_a_position_baselines_have():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.5, "FWD": 0.55}}  # no GKP
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.2, "FWD": 0.25, "GKP": 0.1}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35, "GKP": 0.1}}
    out = trust_gate(model, naive, fpl, full_pipeline=True)
    assert out["trusted"] is False
    assert any("GKP" in f for f in out["failures"])


# --- B15: Tier 2 cannot see the future, and cannot gate trust (2026-09-17 audit) ---

def test_captaincy_is_judged_on_the_squad_you_owned():
    """The old check called it a hit when the globally highest-projected player
    was also the globally highest actual scorer. That is not a captaincy
    decision -- a manager captains someone from his own eleven."""
    pred = {1: pd.Series({10: 9.0, 11: 8.0, 12: 2.0})}
    actual = {1: pd.Series({10: 2.0, 11: 12.0, 12: 30.0})}
    owned = {1: [10, 11]}

    # Globally, 12 is both unowned and the top scorer, so the pool-level check
    # records a miss that says nothing about the decision that was available.
    assert captaincy_hit_rate(pred, actual) == 0.0
    # Among the players actually owned, 11 outscored 10 and the model picked 10.
    assert captaincy_hit_rate(pred, actual, owned_by_gw=owned) == 0.0
    # Pick the right one and it is a hit.
    pred_ok = {1: pd.Series({10: 8.0, 11: 9.0, 12: 2.0})}
    assert captaincy_hit_rate(pred_ok, actual, owned_by_gw=owned) == 1.0


def test_a_component_diagnostic_can_never_grant_production_trust():
    """The rate-only Tier 2 gives the model future playing time and drops every
    nonappearance. Whatever it says, it is not evidence about the pipeline that
    actually picks a team."""
    perfect = {"spearman_by_position": {"MID": 0.9, "DEF": 0.9}}
    weak = {"spearman_by_position": {"MID": 0.1, "DEF": 0.1}}

    full = trust_gate(perfect, weak, weak, full_pipeline=True)
    assert full["trusted"] is True

    component = trust_gate(perfect, weak, weak, full_pipeline=False)
    assert component["trusted"] is False
    assert "component" in component["summary"].lower()


def test_the_trust_gate_still_defaults_to_the_safe_answer():
    """Callers that have not been updated must not accidentally get a pass."""
    perfect = {"spearman_by_position": {"MID": 0.9}}
    weak = {"spearman_by_position": {"MID": 0.1}}
    assert trust_gate(perfect, weak, weak)["trusted"] is False
