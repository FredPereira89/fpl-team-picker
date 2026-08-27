import pandas as pd
import pytest

from fpl.backtest.ledger import (save_predictions, load_predictions,
                                 actuals_from_summaries, score_gameweek, scored_summary,
                                 save_scored_summary, load_scored_summary)

PRED = pd.DataFrame({
    "player_id": [1, 2, 3, 4],
    "web_name": ["Keeper", "Back", "Mid", "Front"],
    "team": ["A", "A", "B", "B"],
    "position": ["GKP", "DEF", "MID", "FWD"],
    "price": [5.0, 5.5, 8.0, 9.0],
    "xp_next1": [4.0, 3.0, 2.0, 1.0],
    "xp_next5": [20.0, 15.0, 10.0, 5.0],
    "p_start": [0.9, 0.8, 0.7, 0.6],
    "e_minutes": [81.0, 72.0, 63.0, 54.0],
    "confidence": ["high"] * 4,
    "flags": [[], [], ["Doubtful: 50% chance of playing"], []],
})


def _summary(pid, rounds):
    return {"history": [{"round": r, "total_points": p, "minutes": m}
                        for r, p, m in rounds]}


def test_predictions_round_trip_through_the_ledger(tmp_path):
    save_predictions(PRED, gw=7, root=tmp_path)
    back = load_predictions(gw=7, root=tmp_path)
    assert list(back.player_id) == [1, 2, 3, 4]
    assert back.loc[back.player_id == 1, "xp_next1"].iloc[0] == 4.0


def test_ledger_records_which_gameweek_it_forecast(tmp_path):
    """A ledger file has to be self-describing — the filename is not enough
    once frames are concatenated for a multi-gameweek report."""
    save_predictions(PRED, gw=7, root=tmp_path)
    assert (load_predictions(gw=7, root=tmp_path)["gw"] == 7).all()


def test_actuals_sum_both_fixtures_of_a_double_gameweek():
    summaries = {1: _summary(1, [(7, 6, 90), (7, 2, 90)]), 2: _summary(2, [(7, 5, 90)])}
    actual = actuals_from_summaries(summaries, gw=7).set_index("player_id")
    assert actual.loc[1, "actual"] == 8
    assert actual.loc[2, "actual"] == 5


def test_actuals_exclude_other_gameweeks():
    summaries = {1: _summary(1, [(6, 12, 90), (7, 2, 90), (8, 9, 90)])}
    actual = actuals_from_summaries(summaries, gw=7).set_index("player_id")
    assert actual.loc[1, "actual"] == 2


def test_score_reports_signed_bias_per_position():
    """Bias must keep its sign — over-prediction is the failure mode the
    goalkeeper audit turned on, and an absolute error would have hidden it."""
    actual = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [1.0, 3.0, 2.0, 1.0],
                           "minutes": [90.0, 90.0, 90.0, 90.0]})
    s = score_gameweek(PRED, actual)
    assert s["bias_by_position"]["GKP"] == pytest.approx(3.0)
    assert s["bias_by_position"]["DEF"] == pytest.approx(0.0)
    assert s["n"] == 4


def test_score_ignores_players_with_no_actual_row():
    actual = pd.DataFrame({"player_id": [1, 2], "actual": [1.0, 3.0],
                           "minutes": [90.0, 90.0]})
    s = score_gameweek(PRED, actual)
    assert s["n"] == 2


def test_score_raises_when_nothing_overlaps():
    """Silently reporting metrics over an empty join would print a confident
    zero for every statistic."""
    actual = pd.DataFrame({"player_id": [99], "actual": [5.0], "minutes": [90.0]})
    with pytest.raises(ValueError, match="no players in common"):
        score_gameweek(PRED, actual)


def test_summary_names_the_positions_the_model_over_predicts():
    actual = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [1.0, 3.0, 2.0, 1.0],
                           "minutes": [90.0] * 4})
    text = scored_summary(score_gameweek(PRED, actual), gw=7)
    assert "GW7" in text
    assert "GKP" in text


def test_scored_summary_persists_for_the_next_run(tmp_path):
    save_scored_summary("GW7 rank quality +0.472", root=tmp_path)
    assert load_scored_summary(root=tmp_path) == "GW7 rank quality +0.472"


def test_no_scored_summary_yet_reads_as_absent(tmp_path):
    assert load_scored_summary(root=tmp_path) is None
