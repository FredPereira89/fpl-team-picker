"""Recalibrating xP against what actually happened.

The 2026-09-09 review found the projection well calibrated POOL-WIDE (bias
-0.08 pts/player-GW) while differing by position -- GKP +0.72, MID -0.16 on
GW2. A position-specific bias is the kind that changes picks, because the
optimizer is choosing between positions under a budget.

A GLOBAL affine recalibration cannot change anything: every squad has the same
fifteen slots, so adding `a` and scaling by `b > 0` leaves the argmax exactly
where it was. That is why this fits per position, and why the no-op test below
is a real property rather than a triviality.
"""
import numpy as np
import pandas as pd
import pytest

from fpl.model.calibration import (fit_calibration, apply_calibration,
                                   Calibration, MIN_GAMEWEEKS)


def _scored(n_gw=6, slope=1.0, intercept=0.0, gk_bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gw + 1):
        for i in range(120):
            pos = ["GKP", "DEF", "MID", "FWD"][i % 4]
            xp = float(rng.uniform(0.2, 8.0))
            mean = intercept + slope * xp + (gk_bias if pos == "GKP" else 0.0)
            rows.append({"gw": gw, "player_id": i, "position": pos, "xp_next1": xp,
                         "actual": float(rng.poisson(max(mean, 0.01)))})
    return pd.DataFrame(rows)


def test_refuses_to_fit_on_too_few_gameweeks():
    """A calibration fitted on two gameweeks is noise dressed as a correction."""
    assert fit_calibration(_scored(n_gw=2)) is None


def test_a_well_calibrated_model_gets_a_near_identity_transform():
    cal = fit_calibration(_scored(n_gw=12, slope=1.0, intercept=0.0))
    assert cal is not None
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert cal.slope[pos] == pytest.approx(1.0, abs=0.25)
        assert cal.intercept[pos] == pytest.approx(0.0, abs=0.5)


def test_an_over_predicted_position_is_scaled_back_relative_to_the_others():
    """GKP xP that buys 0.7 fewer real points per match than it claims must
    come down RELATIVE to the positions that deliver, or the optimizer keeps
    spending budget against a number that is not real."""
    cal = fit_calibration(_scored(n_gw=12, slope=1.0, gk_bias=-0.7))
    xp = pd.DataFrame({"player_id": [1, 2], "position": ["GKP", "MID"],
                       "xp_next1": [5.0, 5.0], "xp_next5": [25.0, 25.0],
                       "xp_horizon": [23.0, 23.0]})
    out = apply_calibration(xp, cal)
    gk, mid = out.set_index("player_id")["xp_next1"]
    assert gk < mid


def test_calibration_moves_every_projection_column_together():
    cal = fit_calibration(_scored(n_gw=12, slope=1.0, gk_bias=-0.7))
    xp = pd.DataFrame({"player_id": [1], "position": ["GKP"], "xp_next1": [5.0],
                       "xp_next5": [25.0], "xp_horizon": [23.0], "xp_gw7": [5.0]})
    out = apply_calibration(xp, cal)
    for col in ("xp_next1", "xp_next5", "xp_horizon", "xp_gw7"):
        assert out[col].iloc[0] != pytest.approx(xp[col].iloc[0])


def test_a_calibrated_projection_is_never_negative():
    cal = fit_calibration(_scored(n_gw=12, slope=1.0, gk_bias=-3.0))
    xp = pd.DataFrame({"player_id": [1], "position": ["GKP"], "xp_next1": [0.2],
                       "xp_next5": [0.4], "xp_horizon": [0.4]})
    assert (apply_calibration(xp, cal)["xp_next1"] >= 0).all()


def test_a_thin_position_is_shrunk_toward_the_pooled_fit():
    """Fitting four separate slopes on a short season overfits. A position with
    almost no observations must not get its own wild slope."""
    df = _scored(n_gw=12, slope=1.0)
    fwd = df[df.position == "FWD"]
    thin = pd.concat([df[df.position != "FWD"], fwd.head(12)])   # ~1 FWD per gameweek
    cal = fit_calibration(thin)
    assert abs(cal.slope["FWD"] - cal.pooled_slope) < abs(cal.slope["MID"] - cal.pooled_slope) + 0.5
    assert cal.n_by_position["FWD"] < cal.n_by_position["MID"]


def test_the_calibration_reports_what_it_was_fitted_on():
    cal = fit_calibration(_scored(n_gw=12))
    assert cal.n_gameweeks == 12
    assert cal.n_observations == 12 * 120
    assert 0.0 <= cal.r2 <= 1.0
    assert "12 gameweeks" in cal.summary


# --- assembling the scored history the fit needs ---------------------------

def test_scored_history_joins_past_forecasts_to_what_happened(tmp_path):
    from fpl.backtest.ledger import save_predictions
    from fpl.model.calibration import scored_history
    pred = pd.DataFrame({"player_id": [1, 2], "web_name": ["A", "B"], "team": ["T", "T"],
                         "position": ["MID", "DEF"], "price": [7.0, 5.0],
                         "xp_next1": [5.0, 3.0], "xp_next5": [25.0, 15.0],
                         "p_start": [0.9, 0.9], "e_minutes": [80.0, 80.0],
                         "confidence": ["high"] * 2, "flags": [[], []]})
    for gw in (1, 2):
        save_predictions(pred, gw=gw, root=tmp_path)
    summaries = {1: {"history": [{"round": 1, "total_points": 9, "minutes": 90},
                                 {"round": 2, "total_points": 2, "minutes": 90}]},
                 2: {"history": [{"round": 1, "total_points": 1, "minutes": 90},
                                 {"round": 2, "total_points": 6, "minutes": 90}]}}
    out = scored_history(tmp_path, summaries, before_event=3)
    assert sorted(out["gw"].unique()) == [1, 2]
    assert len(out) == 4
    assert set(out.columns) >= {"gw", "player_id", "position", "xp_next1", "actual"}


def test_scored_history_excludes_the_gameweek_being_predicted(tmp_path):
    """Fitting on the gameweek you are about to forecast is scoring the model
    on its own answers."""
    from fpl.backtest.ledger import save_predictions
    from fpl.model.calibration import scored_history
    pred = pd.DataFrame({"player_id": [1], "web_name": ["A"], "team": ["T"],
                         "position": ["MID"], "price": [7.0], "xp_next1": [5.0],
                         "xp_next5": [25.0], "p_start": [0.9], "e_minutes": [80.0],
                         "confidence": ["high"], "flags": [[]]})
    for gw in (1, 2, 3):
        save_predictions(pred, gw=gw, root=tmp_path)
    summaries = {1: {"history": [{"round": r, "total_points": 5, "minutes": 90}
                                 for r in (1, 2, 3)]}}
    assert sorted(scored_history(tmp_path, summaries, before_event=3)["gw"].unique()) == [1, 2]


def test_scored_history_skips_gameweeks_nobody_has_played_yet(tmp_path):
    from fpl.backtest.ledger import save_predictions
    from fpl.model.calibration import scored_history
    pred = pd.DataFrame({"player_id": [1], "web_name": ["A"], "team": ["T"],
                         "position": ["MID"], "price": [7.0], "xp_next1": [5.0],
                         "xp_next5": [25.0], "p_start": [0.9], "e_minutes": [80.0],
                         "confidence": ["high"], "flags": [[]]})
    save_predictions(pred, gw=1, root=tmp_path)
    save_predictions(pred, gw=2, root=tmp_path)
    summaries = {1: {"history": [{"round": 1, "total_points": 5, "minutes": 90}]}}
    assert sorted(scored_history(tmp_path, summaries, before_event=9)["gw"].unique()) == [1]
