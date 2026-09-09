"""The distributional layer: points as a DISTRIBUTION, not a point estimate.

Two properties matter and are tested here. First, the simulation must agree
with the analytic model on the MEAN -- they implement the same scoring rules
by different routes, so a disagreement means one of them is wrong. Second, it
must produce the CORRELATION the analytic model cannot express: teammates
share a clean sheet, and that is what makes stacking a real decision rather
than three independent bets.
"""
import numpy as np
import pandas as pd
import pytest

from fpl.config import Config
from fpl.model.simulate import simulate_event, DEFAULT_SIMS
from fpl.model.xp import build_xp

CFG = Config(horizon_gw=1)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5, 6],
    "web_name": ["AlphaGK", "AlphaD1", "AlphaD2", "AlphaFW", "BetaD1", "BetaFW"],
    "team": ["Alpha", "Alpha", "Alpha", "Alpha", "Beta", "Beta"],
    "team_id": [1, 1, 1, 1, 2, 2],
    "position": ["GKP", "DEF", "DEF", "FWD", "DEF", "FWD"],
    "price": [5.0, 5.5, 5.5, 9.0, 5.0, 8.0],
    "selected_by_percent": [10.0, 20.0, 5.0, 40.0, 3.0, 25.0],
})
RATES = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5, 6],
    "xg90": [0.0, 0.08, 0.06, 0.65, 0.05, 0.55],
    "xa90": [0.0, 0.10, 0.09, 0.25, 0.08, 0.20],
    "bonus90": [0.30, 0.35, 0.30, 0.80, 0.25, 0.70],
    "dc90": [0.0, 11.0, 10.0, 2.0, 10.5, 1.5],
    "saves90": [3.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    "cards90": [0.02, 0.20, 0.18, 0.10, 0.22, 0.12],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5, 6],
    "p_start": [0.95, 0.90, 0.75, 0.88, 0.85, 0.80],
    "p_play": [0.96, 0.93, 0.82, 0.92, 0.89, 0.86],
    "p_60": [0.93, 0.84, 0.66, 0.79, 0.77, 0.70],
    "m_start": [90.0, 88.0, 82.0, 79.0, 85.0, 76.0],
    "e_minutes": [85.7, 80.0, 62.9, 70.4, 73.0, 62.0],
    "confidence": ["high"] * 6,
    "flags": [[], [], [], [], [], []],
})
TFX = pd.DataFrame({
    "team_id": [1, 2],
    "event": [1, 1],
    "fixture_id": [1, 1],
    "opponent_id": [2, 1],
    "is_home": [True, False],
    "xgc": [0.85, 1.70],
    "p_cs": [float(np.exp(-0.85)), float(np.exp(-1.70))],
    "att_mult": [1.25, 0.80],
    "opp_threat": [0.7, 1.3],
})
COUNTS = pd.DataFrame([{"team_id": 1, "event": 1, "n_fixtures": 1},
                       {"team_id": 2, "event": 1, "n_fixtures": 1}])


def _sim(n_sims=4000, seed=0):
    return simulate_event(PLAYERS, RATES, MINUTES, TFX, event=1,
                          n_sims=n_sims, seed=seed)


def test_simulation_returns_a_sample_matrix_per_player():
    ids, samples = _sim(n_sims=500)
    assert list(ids) == [1, 2, 3, 4, 5, 6]
    assert samples.shape == (6, 500)


def test_the_same_seed_reproduces_the_same_draws():
    _, a = _sim(n_sims=300, seed=7)
    _, b = _sim(n_sims=300, seed=7)
    assert np.array_equal(a, b)


def test_simulated_mean_agrees_with_the_analytic_projection():
    """The simulation and model.xp implement the same FPL scoring rules by
    different routes -- one in closed form, one by drawing outcomes. If their
    means disagree, one of them has the rules wrong."""
    ids, samples = _sim(n_sims=20000)
    analytic = build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, CFG,
                        from_event=1).set_index("player_id")["xp_next1"]
    for i, pid in enumerate(ids):
        assert samples[i].mean() == pytest.approx(float(analytic.loc[pid]), abs=0.15), \
            f"player {pid}: sim {samples[i].mean():.3f} vs analytic {analytic.loc[pid]:.3f}"


def test_teammates_clean_sheets_are_correlated():
    """Two defenders on the same side keep the same clean sheet. Pricing them
    as independent is what makes the optimizer blind to stacking."""
    ids, samples = _sim(n_sims=8000)
    idx = {p: i for i, p in enumerate(ids)}
    same_team = np.corrcoef(samples[idx[2]], samples[idx[3]])[0, 1]
    cross_team = np.corrcoef(samples[idx[2]], samples[idx[5]])[0, 1]
    assert same_team > 0.15
    assert same_team > cross_team


def test_points_are_a_spread_not_a_point_estimate():
    ids, samples = _sim(n_sims=4000)
    idx = {p: i for i, p in enumerate(ids)}
    striker = samples[idx[4]]
    assert striker.std() > 1.5
    assert striker.max() >= 8          # hauls have to be reachable
    assert (striker == 0).mean() > 0.02  # so do blanks


def test_a_player_with_no_chance_of_playing_never_scores():
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 4, ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    ids, samples = simulate_event(PLAYERS, RATES, mins, TFX, event=1,
                                  n_sims=500, seed=0)
    assert samples[list(ids).index(4)].sum() == 0


def test_a_blank_gameweek_scores_nobody():
    tfx = TFX[TFX.team_id == 1]
    ids, samples = simulate_event(PLAYERS, RATES, MINUTES, tfx, event=1,
                                  n_sims=500, seed=0)
    idx = {p: i for i, p in enumerate(ids)}
    assert samples[idx[5]].sum() == 0   # Beta has no fixture
    assert samples[idx[2]].sum() > 0


def test_a_double_gameweek_scores_twice():
    doubled = pd.concat([TFX, TFX[TFX.team_id == 1].assign(fixture_id=2)])
    ids, one = _sim(n_sims=6000)
    _, two = simulate_event(PLAYERS, RATES, MINUTES, doubled, event=1,
                            n_sims=6000, seed=0)
    idx = {p: i for i, p in enumerate(ids)}
    assert two[idx[2]].mean() == pytest.approx(2 * one[idx[2]].mean(), rel=0.08)
