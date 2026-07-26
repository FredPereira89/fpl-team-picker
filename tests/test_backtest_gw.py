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
    out = trust_gate(model, naive, fpl)
    assert out["trusted"] is True
    assert out["failures"] == []


def test_trust_gate_fails_and_names_the_position():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.1, "FWD": 0.55}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.4, "FWD": 0.25}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}
    out = trust_gate(model, naive, fpl)
    assert out["trusted"] is False
    assert any("DEF" in f for f in out["failures"])
    assert "DEF" in out["summary"]
