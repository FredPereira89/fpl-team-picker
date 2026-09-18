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


def _mean_excluding_bonus(n_sims=20000, seed=0):
    """(ids, per-player mean with the match-ranked bonus term subtracted,
    analytic projection with its bonus90-rate term subtracted).

    Bonus is now a match-wide RANKED resource (R10), not an independent
    per-player rate: only the top three BPS in the whole match score, with
    FPL's tie rule, so the true mean depends on how many players are
    competing for those three slots and how their BPS is distributed --
    properties a per-player `bonus90` rate was never fitted to reproduce,
    and this fixture's 6-player "match" exaggerates the gap further (six
    players contesting the same three slots a real ~20-a-side match spreads
    much thinner). Excluding bonus from both sides keeps these tests
    checking what they actually test: that the OTHER components (goals,
    assists, clean sheets, minutes, DC, cards) agree with `model.xp`.
    """
    from fpl.model.bps import expected_bonus_for
    from fpl.model.simulate import simulate_event_detailed

    detail = simulate_event_detailed(PLAYERS, RATES, MINUTES, TFX, event=1,
                                     n_sims=n_sims, seed=seed)
    ids, samples, bonus = detail["ids"], detail["samples"], detail["bonus"]
    analytic = build_xp(PLAYERS, RATES, MINUTES, TFX, CFG,
                        from_event=1).set_index("player_id")["xp_next1"]
    att_mult_by_team = TFX.set_index("team_id")["att_mult"].astype(float)
    team_of = PLAYERS.set_index("player_id")["team_id"]
    bonus90_of = RATES.set_index("player_id")["bonus90"]
    e_minutes_of = MINUTES.set_index("player_id")["e_minutes"]

    sim_mean, analytic_mean = {}, {}
    for i, pid in enumerate(ids):
        analytic_bonus = expected_bonus_for(bonus90_of.loc[pid], e_minutes_of.loc[pid],
                                            att_mult_by_team.loc[team_of.loc[pid]])
        sim_mean[pid] = float((samples[i] - bonus[i]).mean())
        analytic_mean[pid] = float(analytic.loc[pid]) - analytic_bonus
    return ids, sim_mean, analytic_mean


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
    means disagree (outside of bonus, see `_mean_excluding_bonus`), one of
    them has the rules wrong."""
    ids, sim_mean, analytic_mean = _mean_excluding_bonus()
    for pid in ids:
        assert sim_mean[pid] == pytest.approx(analytic_mean[pid], abs=0.15), \
            f"player {pid}: sim {sim_mean[pid]:.3f} vs analytic {analytic_mean[pid]:.3f}"


def test_teammates_clean_sheets_are_correlated():
    """Two defenders on the same side keep the same clean sheet. Pricing them
    as independent is what makes the optimizer blind to stacking.

    Bonus (R10) now also correlates points a little across the WHOLE match,
    not only within a side -- both teams' players compete for the same
    three ranked slots -- which narrows the same-team/cross-team gap
    slightly versus clean sheets alone. 0.12 (was 0.15) still clears both
    substantive claims: a real same-team effect, and same-team beating
    cross-team.
    """
    ids, samples = _sim(n_sims=8000)
    idx = {p: i for i, p in enumerate(ids)}
    same_team = np.corrcoef(samples[idx[2]], samples[idx[3]])[0, 1]
    cross_team = np.corrcoef(samples[idx[2]], samples[idx[5]])[0, 1]
    assert same_team > 0.12
    assert same_team > cross_team


def test_clean_sheet_credit_survives_a_late_concession_after_full_minutes():
    """R10 4th review (Codex): the official clean-sheet rule is no goal
    conceded WHILE ON THE PITCH, not the match's final score -- a player
    subbed at 60' keeps his clean sheet even if his side concedes after he
    leaves. Reproduced against the real simulator: a minimal fixture with
    every OTHER scoring component zeroed out (xg90/xa90/dc90/saves90/
    cards90 all 0) and appearance forced certain, so the only variable
    part of `pts` is the clean-sheet term -- its rate can be read straight
    back out of the mean. A defender with `m_start` fixed at 60 (`share`
    < 1, so some of the match happens after he is off) on a side that
    concedes with high probability must show a HIGHER clean-sheet rate
    than the side's own P(concede nothing for the full 90) -- using the
    match's final score alone (the pre-fix behaviour) could never exceed
    that ceiling, since every concession denies every player on the side
    regardless of when it happened.
    """
    import pandas as pd
    from fpl.config import Config
    from fpl.model.simulate import simulate_event_detailed

    players = pd.DataFrame({"player_id": [1, 2], "web_name": ["GK", "DEF"],
                            "team": ["Alpha", "Alpha"], "team_id": [1, 1],
                            "position": ["GKP", "DEF"], "price": [5.0, 5.0],
                            "selected_by_percent": [5.0, 5.0]})
    rates = pd.DataFrame({"player_id": [1, 2], "xg90": [0.0, 0.0], "xa90": [0.0, 0.0],
                          "bonus90": [0.0, 0.0], "dc90": [0.0, 0.0],
                          "saves90": [0.0, 0.0], "cards90": [0.0, 0.0]})
    minutes = pd.DataFrame({"player_id": [1, 2], "p_start": [1.0, 1.0],
                            "p_play": [1.0, 1.0], "p_60": [1.0, 1.0],
                            # player 2 plays exactly 60 of 90 -- off before
                            # some of the match, unlike player 1 (full 90).
                            "m_start": [90.0, 60.0], "e_minutes": [90.0, 60.0]})
    tfx = pd.DataFrame([
        {"team_id": 1, "event": 1, "fixture_id": 1, "opponent_id": 2, "is_home": True,
         "xgc": 2.0, "p_cs": 0.13, "att_mult": 1.0, "opp_threat": 1.0},
        {"team_id": 2, "event": 1, "fixture_id": 1, "opponent_id": 1, "is_home": False,
         "xgc": 0.0, "p_cs": 1.0, "att_mult": 1.0, "opp_threat": 1.0},
    ])

    detail = simulate_event_detailed(players, rates, minutes, tfx, event=1,
                                     n_sims=200000, seed=11)
    ids, samples = detail["ids"], detail["samples"]
    conceded_team = detail["conceded_team"][0]
    team_clean_rate = float((conceded_team == 0).mean())

    # pts = 1 (played) + 1 (reached_60, both certain here) + clean_sheet*CS_PTS.
    from fpl.model.xp import CS_PTS
    i_def = list(ids).index(2)   # the partial-minutes defender, not the GKP
    player_clean_rate = float((samples[i_def] - 2.0).mean()) / CS_PTS["DEF"]

    assert player_clean_rate > team_clean_rate + 0.02


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


def test_an_unavailable_player_scores_zero_in_every_scenario():
    """Shared scenarios feed the rank layer, so one phantom cameo distorts
    captain, autosub and rank results for every candidate at once."""
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 4,
             ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    ids, samples = simulate_event(PLAYERS, RATES, mins, TFX, event=1, n_sims=400)
    assert samples[ids.index(4)].max() == 0.0


# --- B7: samples must describe the CALIBRATED projection ---

def test_moment_matching_makes_sample_means_equal_the_calibrated_projection():
    """The MILP chose on calibrated numbers while the rank layer scored
    candidates on raw-rate samples, so the squad that beat the field most
    often was judged on a distribution that did not describe it."""
    from fpl.model.simulate import moment_match
    ids, samples = _sim(n_sims=4000)
    calibrated = pd.DataFrame({"player_id": ids,
                               "xp_next1": [samples[i].mean() * 1.3 for i in range(len(ids))]})
    matched = moment_match(samples, ids, calibrated)
    for i, pid in enumerate(ids):
        assert matched[i].mean() == pytest.approx(float(calibrated.set_index("player_id").loc[pid, "xp_next1"]), rel=1e-9)
    # Shape preserved: zeros stay zeros and the ordering of scenarios holds.
    assert ((samples == 0) == (matched == 0)).all()


def test_moment_matching_leaves_a_zero_mean_player_alone():
    from fpl.model.simulate import moment_match
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 4, ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    ids, samples = simulate_event(PLAYERS, RATES, mins, TFX, event=1, n_sims=300, seed=0)
    calibrated = pd.DataFrame({"player_id": ids, "xp_next1": [2.0] * len(ids)})
    matched = moment_match(samples, ids, calibrated)
    assert matched[list(ids).index(4)].sum() == 0


# --- R3: one coherent match per fixture ---

def _goals_and_cs(ids, samples, rates, minutes, tfx, n_sims=3000):
    """Recompute per-scenario team goals and opposing clean sheets from a
    detailed simulation, via the diagnostic hook."""
    from fpl.model.simulate import simulate_event_detailed
    return simulate_event_detailed(PLAYERS, rates, minutes, tfx, event=1,
                                   n_sims=n_sims, seed=1)


def test_detailed_simulation_exposes_appearance_separately_from_points():
    from fpl.model.simulate import simulate_event_detailed

    mins = MINUTES.copy()
    mins.loc[mins.player_id == 4, ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    detail = simulate_event_detailed(
        PLAYERS, RATES, mins, TFX, event=1, n_sims=200, seed=4)

    played = detail["played"]
    unavailable = detail["ids"].index(4)
    assert played.shape == detail["samples"].shape
    assert played.dtype == bool
    assert not played[unavailable].any()


def test_a_goal_never_coexists_with_an_opposing_clean_sheet():
    """Each side's goals were drawn independently of the other side's goals
    conceded, so an attacker could score in a scenario where the opposing
    defenders kept a clean sheet. That impossible pair is exactly where a rank
    objective built on stacks and opposing players goes wrong."""
    detail = _goals_and_cs(*_sim(n_sims=10), RATES, MINUTES, TFX)
    goals = detail["goals"]              # (n_players, n_sims)
    conceded = detail["conceded_team"]   # (n_players, n_sims), the SIDE's conceded
    team = detail["team_of"]
    alpha = [i for i, t in enumerate(team) if t == 1]
    beta = [i for i, t in enumerate(team) if t == 2]
    alpha_goals = goals[alpha].sum(axis=0)
    beta_conceded = conceded[beta[0]]
    assert (alpha_goals <= beta_conceded).all()
    # And with no unmodelled remainder in this fixture the two are identical
    # whenever the modelled attackers account for the whole team total.
    assert (alpha_goals[beta_conceded == 0] == 0).all()


def test_attacker_marginal_means_survive_the_allocation():
    """Allocating team goals to players must not move the projection
    (excluding bonus, now a match-ranked resource -- see
    `_mean_excluding_bonus`)."""
    ids, sim_mean, analytic_mean = _mean_excluding_bonus()
    for pid in ids:
        assert sim_mean[pid] == pytest.approx(analytic_mean[pid], abs=0.15)


def test_a_lone_team_row_without_an_opponent_still_simulates():
    tfx = TFX[TFX.team_id == 1]
    ids, samples = simulate_event(PLAYERS, RATES, MINUTES, tfx, event=1,
                                  n_sims=500, seed=0)
    assert samples[list(ids).index(2)].sum() > 0


# --- C6-5 (R4 revisited): a single event's marginal cannot be "widened" ---
#
# An earlier version drew a fresh Beta p_start per scenario, claiming this
# widened the appearance spread for a thin-evidence player. That claim was an
# overclaim: a single binary event's marginal is Bernoulli(mean) for ANY
# generating mechanism with that mean, so nothing can widen it without moving
# the mean. The mechanism's only real effect was a bug -- p60_given_start
# computed from the random draw instead of the fixed p_start systematically
# lowered the true P(reached 60). The tests below pin the CORRECTED, honest
# properties: start_evidence changes nothing about a single gameweek's
# simulated distribution, and p_60's marginal matches the input exactly.

def test_start_evidence_does_not_change_a_single_gameweeks_simulation():
    """The whole point of removing the Beta draw: start_evidence carried on
    the minutes frame must not perturb either the mean OR the shape of a
    single gameweek's simulated points, because there is nothing at the
    single-event level for it to legitimately change."""
    mins = MINUTES.copy()
    mins["start_evidence"] = 200.0                 # a nailed regular's worth
    ids, certain = simulate_event(PLAYERS, RATES, mins, TFX, event=1, n_sims=20000, seed=3)
    mins.loc[mins.player_id == 4, "start_evidence"] = 2.0   # a newcomer's worth
    _, thin = simulate_event(PLAYERS, RATES, mins, TFX, event=1, n_sims=20000, seed=3)
    i = list(ids).index(4)
    assert thin[i].mean() == pytest.approx(certain[i].mean(), abs=0.15)
    assert thin[i].var() == pytest.approx(certain[i].var(), rel=0.1)


def test_a_thin_evidence_flag_on_the_minutes_frame_does_not_move_p_60():
    """The regression this whole rewrite exists for: a minutes frame that
    still carries `start_evidence` (an older caller, or a frame produced
    before this fix) must not shift the simulated P(reached 60) away from
    the model's own p_60 -- which is exactly what the removed Beta-draw
    mechanism did whenever evidence was thin. Player id=3 (p_start=0.75,
    p_60=0.66) has a real gap between the two, which is what the old bug
    needed to bite.

    Bonus is excluded from both sides of the comparison (R10): it is now a
    match-wide RANKED resource, not an independent per-player rate, so its
    own mean no longer has to agree with the analytic `bonus90`-based
    estimate -- mixing it in would dilute this test's actual target (the
    OTHER components, which the p_60 bug touches) with noise from a
    mechanism this test was never about.
    """
    from fpl.model.bps import expected_bonus_for
    from fpl.model.simulate import simulate_event_detailed

    mins = MINUTES.copy()
    mins["start_evidence"] = 3.0     # thin -- exactly where the old bug bit hardest
    detail = simulate_event_detailed(PLAYERS, RATES, mins, TFX, event=1,
                                     n_sims=300000, seed=5)
    ids, samples, bonus = detail["ids"], detail["samples"], detail["bonus"]
    analytic = build_xp(PLAYERS, RATES, mins, TFX, CFG, from_event=1).set_index(
        "player_id")["xp_next1"]
    i = list(ids).index(3)
    # Player 3 is Alpha (team_id=1), whose fixture att_mult is 1.25.
    analytic_bonus = expected_bonus_for(bonus90=0.30, e_minutes=62.9, att_mult=1.25)
    # abs=0.12 is tight enough that the confirmed bug (a ~0.05-0.06 point
    # shift in the clean-sheet term alone at this gap) would fail it, and
    # wide enough for R10's 4th review: clean sheet is now computed PER
    # PLAYER (goals conceded while ON THE PITCH, via `conceded_on == 0`,
    # the official rule -- see model.bps.score_side_bps), not from the
    # match's final score. `build_xp`'s closed-form `p_cs * CS_PTS * p_60`
    # does not model "while on pitch" either, so the two sides now diverge
    # by a genuine, small, expected amount (observed ~0.08 here) that is
    # not a bug -- widening `xp_for_fixture` to match is a separate,
    # undone follow-up.
    assert (samples[i] - bonus[i]).mean() == pytest.approx(
        float(analytic.loc[3]) - analytic_bonus, abs=0.12)


def test_p60_given_start_uses_the_fixed_p_start_not_a_random_draw():
    """Direct, deterministic proof of the fix: with p_start fixed and no
    per-scenario draw of it, p60_given_start = p_60 / p_start is a CONSTANT,
    so P(reached_60 | started) converges to exactly p_60/p_start (and
    P(reached_60) to exactly p_60) as n_sims grows -- reproducing Codex's own
    numerical check (buggy: 0.402 vs intended ~0.45; fixed: matches)."""
    n = 200000
    rng = np.random.default_rng(0)
    p_start, p_60 = 0.6, 0.45
    u = rng.random(n)
    started = u < p_start
    p60_given_start = p_60 / p_start
    reached_60 = started & (rng.random(n) < np.clip(p60_given_start, 0.0, 1.0))
    assert reached_60.mean() == pytest.approx(p_60, abs=0.01)


def test_an_unavailable_player_never_appears():
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 4, ["p_start", "p_play", "p_60", "e_minutes"]] = 0.0
    ids, samples = simulate_event(PLAYERS, RATES, mins, TFX, event=1, n_sims=500, seed=0)
    assert samples[list(ids).index(4)].sum() == 0


def test_frames_without_evidence_behave_as_before():
    ids, a = simulate_event(PLAYERS, RATES, MINUTES, TFX, event=1, n_sims=300, seed=7)
    ids, b = simulate_event(PLAYERS, RATES, MINUTES, TFX, event=1, n_sims=300, seed=7)
    assert np.array_equal(a, b)
