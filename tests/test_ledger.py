import numpy as np
import pandas as pd
import pytest

from fpl.backtest.ledger import (save_predictions, load_predictions,
                                 actuals_from_summaries, score_gameweek, scored_summary,
                                 save_scored_summary, load_scored_summary,
                                 common_pool, spearman_on, compare_rankers,
                                 spearman_ci, probability_scores)

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


def test_score_reports_rank_quality_inside_the_candidate_pool():
    """Pool-wide Spearman is dominated by separating starters from reserves,
    which no manager needs help with. Every real decision is taken among the
    handful of players the model already rates highest, so skill has to be
    reported there too — GW2 scored +0.595 overall and +0.114 in its own top 60.
    """
    actual = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [1.0, 2.0, 3.0, 4.0],
                           "minutes": [90.0] * 4})
    s = score_gameweek(PRED, actual, top_n=3)
    # The model ranked 1 > 2 > 3; actuals ran the other way.
    assert s["spearman_top_n"] == pytest.approx(-1.0)
    assert s["n_top"] == 3


def test_bias_is_split_by_whether_the_player_actually_appeared():
    """A position can look over-predicted purely because players who never
    featured were given points — a minutes failure, not a scoring one. GW2's
    goalkeeper warning was entirely this: bias -0.02 among keepers who started,
    +1.03 among those who did not."""
    actual = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [4.0, 0.0, 2.0, 1.0],
                           "minutes": [90.0, 0.0, 90.0, 90.0]})
    s = score_gameweek(PRED, actual)
    assert s["bias_played"] == pytest.approx(0.0)      # GKP 4.0 vs 4.0, etc.
    assert s["bias_absent"] == pytest.approx(3.0)      # DEF predicted 3.0, never played


def test_summary_blames_minutes_not_scoring_when_absentees_drive_the_bias():
    actual = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [4.0, 0.0, 2.0, 1.0],
                           "minutes": [90.0, 0.0, 90.0, 90.0]})
    text = scored_summary(score_gameweek(PRED, actual), gw=7)
    assert "did not play" in text


def test_scored_summary_persists_for_the_next_run(tmp_path):
    save_scored_summary("GW7 rank quality +0.472", root=tmp_path)
    assert load_scored_summary(root=tmp_path) == "GW7 rank quality +0.472"


def test_no_scored_summary_yet_reads_as_absent(tmp_path):
    assert load_scored_summary(root=tmp_path) is None


# --- Reproducible forecast versions (2026-09-07 review) ---

def test_every_write_keeps_an_immutable_copy(tmp_path):
    """Overwriting gw{n}.parquet destroyed the forecast the optimizer actually
    acted on the moment anything was re-run -- including the pre-deadline one a
    mid-week team-news update replaced."""
    from datetime import datetime, timezone
    from fpl.backtest.ledger import forecast_versions
    first = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    second = datetime(2026, 9, 2, 10, tzinfo=timezone.utc)
    save_predictions(PRED, gw=4, root=tmp_path, created_at=first)
    revised = PRED.copy()
    revised["xp_next1"] = revised["xp_next1"] + 1.0
    save_predictions(revised, gw=4, root=tmp_path, created_at=second)

    versions = forecast_versions(4, tmp_path)
    assert len(versions) == 2
    earlier = pd.read_parquet(versions[0])
    assert earlier["xp_next1"].tolist() == PRED["xp_next1"].tolist()
    # the canonical file is the latest, which is what the scorer reads
    assert load_predictions(4, tmp_path)["xp_next1"].tolist() == revised["xp_next1"].tolist()


def test_a_forecast_records_when_and_under_what_settings_it_was_made(tmp_path):
    from fpl.config import Config
    from fpl.backtest.ledger import MODEL_VERSION, config_fingerprint
    cfg = Config(horizon_gw=5)
    save_predictions(PRED, gw=4, root=tmp_path, cfg=cfg,
                     sources={"bootstrap-static": "2026-09-01T10:00:00Z"})
    row = load_predictions(4, tmp_path).iloc[0]
    assert row["model_version"] == MODEL_VERSION
    assert row["config_hash"] == config_fingerprint(cfg)
    assert "bootstrap-static" in row["sources"]
    assert row["created_at"]


def test_the_same_settings_fingerprint_the_same_way():
    from fpl.config import Config
    from fpl.backtest.ledger import config_fingerprint
    assert config_fingerprint(Config()) == config_fingerprint(Config())
    assert config_fingerprint(Config(horizon_gw=5)) != config_fingerprint(Config(horizon_gw=3))


def test_a_gameweek_is_final_only_once_fpl_has_checked_every_fixture():
    """`finished_provisional` flips at the whistle; bonus and stat corrections
    land after `finished`."""
    from fpl.backtest.ledger import gameweek_is_final
    played = [{"event": 3, "finished": True, "finished_provisional": True},
              {"event": 3, "finished": True, "finished_provisional": True}]
    whistle = [{"event": 3, "finished": True, "finished_provisional": True},
               {"event": 3, "finished": False, "finished_provisional": True}]
    assert gameweek_is_final(played, 3) is True
    assert gameweek_is_final(whistle, 3) is False
    assert gameweek_is_final([], 3) is False


def test_a_provisional_score_says_so_at_the_top():
    """The weekly report quotes this verdict verbatim, so an unfinished
    gameweek must not read as a settled measurement."""
    actuals = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [6.0, 2.0, 8.0, 1.0],
                            "minutes": [90, 90, 90, 0]})
    scored = score_gameweek(PRED, actuals)
    assert scored_summary(scored, 3, provisional=True).startswith("PROVISIONAL")
    assert not scored_summary(scored, 3).startswith("PROVISIONAL")


# --- comparing rankers fairly (2026-09-09 audit) --------------------------
# `spearman_top_n` selects the top N BY THE MODEL and then correlates the
# model's own score inside that truncated set. Selecting on the predictor
# collapses its variance while the outcome's stays wide, so the correlation
# is attenuated by construction -- and, because every ranker would be scored
# on a different set of players, it cannot be compared against a baseline.

def _skewed(n=200, noise=1.0, seed=0):
    """A pool where the model has real, uniform skill across the whole range."""
    import numpy as np
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 2, n)
    return pd.DataFrame({
        "player_id": range(n),
        "position": ["MID"] * n,
        "price": np.round(4.0 + truth * 0.4 + rng.normal(0, 0.3, n), 1),
        "xp_next1": truth + rng.normal(0, noise * 0.3, n),
        "actual": truth + rng.normal(0, noise, n),
        "minutes": 90.0,
    })


def test_common_pool_is_the_union_of_each_rankers_top_n():
    """Every candidate ranker must be scored on the SAME players, or the
    comparison measures which pool each one happened to pick."""
    df = _skewed()
    pool = common_pool(df, ["xp_next1", "price"], top_n=20)
    top_model = set(df.nlargest(20, "xp_next1")["player_id"])
    top_price = set(df.nlargest(20, "price")["player_id"])
    assert set(pool["player_id"]) == top_model | top_price


def test_scoring_a_ranker_on_its_own_top_n_understates_its_skill():
    """The bug this metric had: a model with genuine skill scores far lower on
    the pool it selected than on a pool selected independently of it."""
    df = _skewed()
    own = spearman_on(df.nlargest(40, "xp_next1"), "xp_next1")
    fair = spearman_on(common_pool(df, ["xp_next1", "price"], top_n=40), "xp_next1")
    assert own < fair


def test_ranker_comparison_scores_every_candidate_on_one_pool():
    df = _skewed()
    out = compare_rankers(df, ["xp_next1", "price"], top_n=40)
    assert set(out) == {"xp_next1", "price"}
    assert out["xp_next1"]["n"] == out["price"]["n"]
    # The model is built to beat price here; the comparison has to show it.
    assert out["xp_next1"]["rho"] > out["price"]["rho"]


def test_a_correlation_is_reported_with_a_confidence_interval():
    df = _skewed()
    rho, lo, hi = spearman_ci(df, "xp_next1", n_boot=300, seed=1)
    assert lo < rho < hi


def test_a_small_pool_gets_a_visibly_wider_interval():
    """60 players is not enough to distinguish 'no skill' from 'good skill',
    and the verdict has to say so rather than print a bare number."""
    df = _skewed(n=400)
    _, lo_big, hi_big = spearman_ci(df, "xp_next1", n_boot=300, seed=1)
    _, lo_sm, hi_sm = spearman_ci(df.head(30), "xp_next1", n_boot=300, seed=1)
    assert (hi_sm - lo_sm) > (hi_big - lo_big)


def test_score_gameweek_compares_the_model_against_baselines_fairly():
    scored = score_gameweek(PRED, pd.DataFrame({
        "player_id": [1, 2, 3, 4], "actual": [2.0, 6.0, 9.0, 1.0],
        "minutes": [90.0, 90.0, 90.0, 90.0]}))
    assert "ranker_comparison" in scored
    assert set(scored["ranker_comparison"]) >= {"xp_next1", "price"}
    ns = {v["n"] for v in scored["ranker_comparison"].values()}
    assert len(ns) == 1, "every ranker must be scored on the same pool"


def test_summary_reports_the_interval_and_the_fair_comparison():
    scored = score_gameweek(PRED, pd.DataFrame({
        "player_id": [1, 2, 3, 4], "actual": [2.0, 6.0, 9.0, 1.0],
        "minutes": [90.0, 90.0, 90.0, 90.0]}))
    text = scored_summary(scored, gw=7)
    assert "95%" in text
    assert "price" in text


def test_summary_warns_that_one_gameweek_settles_nothing():
    scored = score_gameweek(PRED, pd.DataFrame({
        "player_id": [1, 2, 3, 4], "actual": [2.0, 6.0, 9.0, 1.0],
        "minutes": [90.0, 90.0, 90.0, 90.0]}))
    assert "one gameweek" in scored_summary(scored, gw=7).lower()


# --- B14: the ledger serves the forecast that was ACTED ON (2026-09-17 audit) ---

def _frame():
    return pd.DataFrame({
        "player_id": [1, 2], "position": ["MID", "DEF"],
        "web_name": ["A", "B"], "price": [8.0, 5.0],
        "xp_next1": [5.0, 3.0], "p_start": [0.9, 0.8],
    })


def test_a_replay_write_does_not_change_what_the_ledger_serves(tmp_path):
    """A replay is built from today's prices, status and news. Letting it
    become the scored record of a live gameweek measures a model that had
    information the live one did not."""
    from fpl.backtest.ledger import save_predictions, load_predictions

    live = _frame()
    save_predictions(live, 5, tmp_path, origin="live",
                     deadline="2026-09-11T17:30:00Z")

    replay = _frame()
    replay["xp_next1"] = [99.0, 99.0]
    save_predictions(replay, 5, tmp_path, origin="replay")

    served = load_predictions(5, tmp_path)
    assert served["xp_next1"].max() == 5.0


def test_a_later_rerun_does_not_displace_the_actioned_forecast(tmp_path):
    """Both writes are pre-deadline here (the deadline is far in the future);
    the point is that an actioned marker beats recency."""
    from fpl.backtest.ledger import save_predictions, load_predictions
    from fpl.backtest.manifest import mark_actioned

    deadline = "2099-01-01T00:00:00Z"
    save_predictions(_frame(), 5, tmp_path, origin="live", deadline=deadline)
    assert mark_actioned(tmp_path, gw=5, deadline=deadline) is not None

    later = _frame()
    later["xp_next1"] = [1.0, 1.0]
    save_predictions(later, 5, tmp_path, origin="live", deadline=deadline)

    assert load_predictions(5, tmp_path)["xp_next1"].max() == 5.0


def test_a_gameweek_with_only_a_replay_is_not_served_at_all(tmp_path):
    from fpl.backtest.ledger import save_predictions, load_predictions

    save_predictions(_frame(), 5, tmp_path, origin="replay")
    with pytest.raises(FileNotFoundError):
        load_predictions(5, tmp_path)


def test_a_ledger_written_before_the_manifest_still_loads(tmp_path):
    """Backwards compatibility: gw{n}.parquet files exist from before this."""
    from fpl.backtest.ledger import LEDGER_DIR, load_predictions

    out = tmp_path / LEDGER_DIR
    out.mkdir(parents=True)
    frame = _frame()
    frame.insert(0, "gw", 5)
    frame.to_parquet(out / "gw5.parquet", index=False)
    assert load_predictions(5, tmp_path)["xp_next1"].max() == 5.0


def test_calibration_skips_a_gameweek_whose_only_forecast_is_a_replay(tmp_path):
    from fpl.backtest.ledger import save_predictions
    from fpl.model.calibration import scored_history

    save_predictions(_frame(), 5, tmp_path, origin="replay")
    summaries = {1: {"history": [{"round": 5, "total_points": 6, "minutes": 90}]},
                 2: {"history": [{"round": 5, "total_points": 2, "minutes": 90}]}}
    assert len(scored_history(tmp_path, summaries, before_event=6)) == 0


# --- proper probability scoring ---

def test_brier_rewards_calibrated_probabilities():
    from fpl.backtest.ledger import brier
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0, 1, 4000))
    y = pd.Series((rng.uniform(0, 1, 4000) < p).astype(float))     # calibrated
    good = brier(p, y)
    bad = brier(pd.Series(1 - p.to_numpy()), y)                       # inverted
    assert good["brier"] < good["climatology"] < bad["brier"]
    assert good["skill"] > 0.3
    for row in good["reliability"]:
        assert abs(row["forecast"] - row["observed"]) < 0.08


def test_probability_scores_settle_appearance_and_the_hour():
    from fpl.backtest.ledger import probability_scores
    df = pd.DataFrame({"p_play": [0.9, 0.9, 0.1, 0.1], "p_60": [0.8, 0.2, 0.05, 0.05],
                       "minutes": [90, 30, 0, 0]})
    out = probability_scores(df)
    assert out["p_play"]["n"] == 4 and out["p_60"]["n"] == 4
    assert out["p_play"]["brier"] < 0.05
    assert "p_start" not in out


def test_a_scored_gameweek_carries_its_probability_scores():
    from fpl.backtest.ledger import score_gameweek
    pred = PRED.copy()
    pred["p_play"] = [0.95, 0.9, 0.9, 0.9]
    pred["p_60"] = [0.9, 0.8, 0.8, 0.8]
    actuals = pd.DataFrame({"player_id": [1, 2, 3, 4], "actual": [6.0, 2.0, 8.0, 1.0],
                            "minutes": [90.0, 90.0, 75.0, 0.0]})
    scored = score_gameweek(pred, actuals)
    assert "p_play" in scored["probability"]
    assert scored["probability"]["p_play"]["n"] == 4


# --- C6-4: p_play/p_60 are single-fixture; a double must not be scored on them ---

def test_actuals_from_summaries_reports_the_fixture_count():
    """`fixture_count` is the number of history rows matched for the round --
    already computed on the way to summing them, just discarded."""
    summaries = {1: _summary(1, [(7, 6, 90), (7, 2, 90)]), 2: _summary(2, [(7, 5, 90)])}
    actual = actuals_from_summaries(summaries, gw=7).set_index("player_id")
    assert actual.loc[1, "fixture_count"] == 2
    assert actual.loc[2, "fixture_count"] == 1


def test_probability_scores_exclude_double_gameweek_rows():
    """p_play/p_60 are single-fixture probabilities: FPL's appearance and
    60-minute bonus are both PER MATCH. Comparing a single-match forecast
    against gameweek-TOTAL minutes systematically understates a double's true
    appearance chance, and cannot tell '60 in one match of two' from a
    35+35 split that reached 60 in neither -- so a double must not be scored
    on either probability, only on the points/error metrics that already sum
    correctly across fixtures."""
    df = pd.DataFrame({
        "player_id": [1, 2],
        "p_play": [0.85, 0.9], "p_60": [0.7, 0.8],
        "minutes": [135.0, 90.0],        # player 1: a double, 90+45
        "fixture_count": [2, 1],
    })
    out = probability_scores(df)
    assert out["p_play"]["n"] == 1
    assert out["p_60"]["n"] == 1


def test_probability_scores_without_a_fixture_count_column_score_everyone():
    """Backward compatible: callers that never learned about fixture_count
    (or a genuinely single-fixture-only frame) keep scoring every row."""
    df = pd.DataFrame({"player_id": [1, 2], "p_play": [0.85, 0.9],
                       "p_60": [0.7, 0.8], "minutes": [90.0, 45.0]})
    out = probability_scores(df)
    assert out["p_play"]["n"] == 2


def test_score_gameweek_carries_fixture_count_through_to_probability_scores():
    """The merge in score_gameweek must actually deliver fixture_count into
    probability_scores, not just have it available on actuals."""
    pred = PRED.copy()
    pred["p_play"] = [0.95, 0.9, 0.9, 0.9]
    pred["p_60"] = [0.9, 0.8, 0.8, 0.8]
    actuals = pd.DataFrame({
        "player_id": [1, 2, 3, 4], "actual": [6.0, 2.0, 8.0, 1.0],
        "minutes": [180.0, 90.0, 75.0, 0.0], "fixture_count": [2, 1, 1, 1],
    })
    scored = score_gameweek(pred, actuals)
    assert scored["probability"]["p_play"]["n"] == 3   # player 1 (a double) excluded
