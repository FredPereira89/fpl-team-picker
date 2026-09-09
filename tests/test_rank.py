"""Rank utility: the objective that actually matches "win the league".

Expected points is the right objective for a game scored on points. FPL is
scored on RANK, and the two come apart in one specific way: points you score
that the field also scores move you nowhere. So a rival field is drawn from
the SAME simulated matches as your own squad -- if you and the field both own
the striker, his haul appears on both sides and cancels, which is exactly the
effect an expected-points objective cannot see.
"""
import numpy as np
import pandas as pd
import pytest

from fpl.optimize.rank import (sample_rival_squads, squad_scores, rank_percentile,
                               best_captain_by_rank, p_beat_target, field_weights,
                               RIVALS)

# 40 players over 8 clubs: enough for legal XIs, small enough to reason about.
POOL = pd.DataFrame([
    {"player_id": 100 + i,
     "web_name": f"P{i}",
     "team": f"T{i % 8}",
     "team_id": i % 8,
     "position": ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i % 8],
     "price": 4.0 + (i % 7) * 0.9,
     "xp_next1": 2.0 + (i % 9) * 0.35,
     "p_start": 0.9,
     "p_play": 0.93,
     # Half the pool is template, half is invisible.
     "ownership": 45.0 if i % 2 == 0 else 0.6}
    for i in range(40)
])
IDS = [int(i) for i in POOL["player_id"]]


def _samples(n_sims=1500, seed=0):
    """Points draws with the right shape: mean tracks xp, spread is real."""
    rng = np.random.default_rng(seed)
    mean = POOL["xp_next1"].to_numpy()[:, None]
    return rng.poisson(np.broadcast_to(mean, (len(POOL), n_sims))).astype(float)


def test_rival_squads_are_legal_starting_elevens():
    rng = np.random.default_rng(0)
    rivals = sample_rival_squads(POOL, n_rivals=50, rng=rng)
    assert rivals.shape == (50, len(POOL))
    pos = POOL["position"].to_numpy()
    for row in rivals:
        picked = row > 0
        assert picked.sum() == 11
        assert (pos[picked] == "GKP").sum() == 1
        assert 3 <= (pos[picked] == "DEF").sum() <= 5
        assert 1 <= (pos[picked] == "FWD").sum() <= 3
        assert row.max() == 2      # exactly one captain, counted twice
        assert row.sum() == 12


def test_rivals_are_drawn_toward_the_template():
    """The field owns what the field owns. A rival squad sampled uniformly
    would make every differential look free."""
    rng = np.random.default_rng(0)
    rivals = sample_rival_squads(POOL, n_rivals=400, rng=rng)
    owned = POOL["ownership"].to_numpy()
    picked_rate = (rivals > 0).mean(axis=0)
    assert picked_rate[owned > 10].mean() > 3 * picked_rate[owned < 10].mean()


def test_a_squad_of_the_most_owned_players_lands_mid_table():
    """Own exactly what everyone owns and you finish level with everyone: the
    definition of average, and the trap an expected-points objective walks
    into when the template is also the highest-xP team."""
    samples = _samples()
    rng = np.random.default_rng(1)
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=rng)
    rival_scores = squad_scores(rivals, samples)
    template = sample_rival_squads(POOL, n_rivals=1, rng=np.random.default_rng(1))[0]
    pct = rank_percentile(squad_scores(template[None, :], samples)[0], rival_scores)
    assert 0.30 < pct < 0.70


def test_a_genuinely_better_squad_ranks_above_the_field():
    samples = _samples()
    rng = np.random.default_rng(1)
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=rng)
    rival_scores = squad_scores(rivals, samples)
    # Hand-build the best legal XI by xp, captain the top scorer.
    best = np.zeros(len(POOL))
    order = POOL.sort_values("xp_next1", ascending=False)
    quota = {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1}
    for _, r in order.iterrows():
        if quota.get(r["position"], 0) > 0:
            best[IDS.index(int(r["player_id"]))] = 1
            quota[r["position"]] -= 1
    best[int(np.argmax(np.where(best > 0, POOL["xp_next1"], -1)))] = 2
    pct = rank_percentile(squad_scores(best[None, :], samples)[0], rival_scores)
    assert pct > 0.75


def _captaincy_case(n=40000, spread=2.0, seed=0):
    """A tight squad plus two armband options with the SAME mean of 6.0: one
    that always returns 6, one that returns 0 or 20."""
    rng = np.random.default_rng(seed)
    base = rng.normal(33.0, spread, n)
    steady = np.full(n, 6.0)
    spiky = rng.choice([0.0, 20.0], n, p=[0.7, 0.3])
    return base, steady, spiky


def test_variance_does_not_buy_a_better_average_rank():
    """The folk rule "captain the ceiling" is false for EXPECTED rank: a 70%
    blank costs more than a 30% haul buys, at identical means."""
    base, steady, spiky = _captaincy_case()
    rivals = np.random.default_rng(1).normal(38.0, 6.0, (RIVALS, len(base)))
    assert p_beat_target(base + steady, rivals) > p_beat_target(base + spiky, rivals)


def test_variance_wins_once_the_target_is_out_of_reach_without_it():
    """The real trade. If the steady option cannot get you to the bar, the
    volatile one is the only thing that can -- and the crossover is sharp."""
    base, steady, spiky = _captaincy_case()
    reachable = np.full((RIVALS, len(base)), 36.0)     # steady clears this
    unreachable = np.full((RIVALS, len(base)), 45.0)   # only a haul clears this
    assert p_beat_target(base + steady, reachable) > p_beat_target(base + spiky, reachable)
    assert p_beat_target(base + spiky, unreachable) > p_beat_target(base + steady, unreachable)


def test_the_better_armband_flips_as_the_target_rises():
    """Not a smooth premium -- a sign change. The steady captain wins while the
    target is reachable without a haul and loses the moment it is not. (The
    ceiling's edge then FADES again at extreme targets, once even the haul
    stops clearing the bar, so this is deliberately not asserted as monotone.)"""
    base, steady, spiky = _captaincy_case()
    rivals = np.random.default_rng(1).normal(38.0, 6.0, (RIVALS, len(base)))
    gap = lambda t: (p_beat_target(base + steady, rivals, t)
                     - p_beat_target(base + spiky, rivals, t))
    assert gap(0.5) > 0     # beating the median: take the sure thing
    assert gap(0.9) < 0     # top-tenth week: only the ceiling gets there
    assert gap(0.99) < 0


def test_captaincy_by_rank_takes_the_better_player_at_the_default_target():
    n = 4000
    rng = np.random.default_rng(0)
    xi = IDS[:11]
    samples = rng.poisson(2.0, (len(POOL), n)).astype(float)
    good, ordinary = IDS.index(xi[0]), IDS.index(xi[1])
    samples[good] = rng.poisson(7.0, n)
    samples[ordinary] = rng.poisson(3.0, n)
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=np.random.default_rng(2))
    rival_scores = squad_scores(rivals, samples)
    assert best_captain_by_rank(xi, IDS, samples, rival_scores) == xi[0]


def test_field_weights_describe_about_one_team_plus_an_armband():
    w = field_weights(POOL)
    assert w.sum() == pytest.approx(12.0, rel=0.02)   # 11 starters + 1 captain
    assert (w >= 0).all()


# --- choosing between candidate squads -------------------------------------

def _fake_squad(ids_in, xi, xp=0.0):
    from fpl.optimize.squad import Squad
    return Squad(player_ids=list(ids_in), starting_ids=list(xi),
                 total_cost=100.0, xp=xp)


def test_scoring_a_candidate_reports_where_it_lands_against_the_field():
    from fpl.optimize.rank import score_candidate
    samples = _samples()
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=np.random.default_rng(1))
    rival_scores = squad_scores(rivals, samples)
    squad = _fake_squad(IDS[:15], IDS[:11])
    out = score_candidate(squad, IDS, samples, rival_scores)
    assert out["captain"] in IDS[:11]
    assert 0.0 <= out["p_beat_target"] <= 1.0
    assert 0.0 <= out["rank_percentile"] <= 1.0
    assert out["mean_points"] > 0


def test_the_chosen_squad_is_the_one_that_beats_the_field_most_often():
    from fpl.optimize.rank import pick_best_squad
    samples = _samples()
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=np.random.default_rng(1))
    rival_scores = squad_scores(rivals, samples)
    strong = POOL.nlargest(15, "xp_next1")["player_id"].astype(int).tolist()
    weak = POOL.nsmallest(15, "xp_next1")["player_id"].astype(int).tolist()
    candidates = [_fake_squad(weak, weak[:11]), _fake_squad(strong, strong[:11])]
    best, scored = pick_best_squad(candidates, IDS, samples, rival_scores)
    assert set(best.player_ids) == set(strong)
    assert len(scored) == 2


def test_an_empty_candidate_list_is_refused_rather_than_returning_nothing():
    from fpl.optimize.rank import pick_best_squad
    with pytest.raises(ValueError, match="no candidate"):
        pick_best_squad([], IDS, _samples(), np.zeros((RIVALS, 1500)))


def test_rival_pick_rates_track_actual_ownership():
    """The field has to be as strong as the real field, or every squad looks
    better than it is. Weighted draws WITHOUT replacement distort the marginal:
    a 71%-owned striker competing for three forward slots against a long tail
    of forwards came out in 47% of rival XIs, so the simulated field scored
    ~44 against a real FPL average nearer 50-55.

    Systematic PPS sampling fixes the marginals by construction.
    """
    # A realistic shape: one 71%-owned striker against a long tail of ~70 other
    # forwards. The tail is the whole problem -- with three forward slots and
    # weights proportional to ownership, the tail crowds him out.
    rows = []
    pid = 0
    for pos, n in (("GKP", 30), ("DEF", 90), ("MID", 100), ("FWD", 70)):
        for j in range(n):
            rows.append({"player_id": pid, "web_name": f"{pos}{j}", "team": f"T{pid % 20}",
                         "team_id": pid % 20, "position": pos, "price": 5.0,
                         "xp_next1": 3.0, "p_start": 0.9, "p_play": 0.93,
                         "ownership": 5.0})
            pid += 1
    pool = pd.DataFrame(rows)
    # Highly owned because he is good -- which is why owners start him.
    star = pool.index[pool.position == "FWD"][0]
    pool.loc[star, "ownership"] = 71.0
    pool.loc[star, "xp_next1"] = 8.8
    rivals = sample_rival_squads(pool, n_rivals=3000, rng=np.random.default_rng(0))
    rate = (rivals[:, star] > 0).mean()
    assert rate > 0.60, f"dominant striker appeared in only {rate:.0%} of XIs"


def test_every_rival_xi_is_still_exactly_eleven_with_one_captain():
    pool = POOL.copy()
    pool.loc[pool.index[0], "ownership"] = 99.0
    rivals = sample_rival_squads(pool, n_rivals=200, rng=np.random.default_rng(3))
    assert ((rivals > 0).sum(axis=1) == 11).all()
    assert (rivals.max(axis=1) == 2).all()
    assert (rivals.sum(axis=1) == 12).all()


def test_a_player_never_starts_for_more_managers_than_own_him():
    """Ownership is a hard ceiling on the XI rate: you cannot field a player
    you did not buy. Weighting slots by ownership x projection pushed a
    71%-owned striker to 85% of rival XIs without this cap."""
    # Ownership per position sums to the fifteen a manager actually owns
    # (2 GKP, 5 DEF, 5 MID, 3 FWD), so filling every slot under the ceilings is
    # possible -- as it is in real data.
    rows = []
    for pid, (pos, own, xp_) in enumerate(
            [("GKP", 10.0, 4.0)] * 20 + [("DEF", 8.3, 3.0)] * 60
            + [("MID", 8.3, 3.0)] * 60 + [("FWD", 6.15, 2.0)] * 40):
        rows.append({"player_id": pid, "web_name": f"P{pid}", "team": f"T{pid % 20}",
                     "team_id": pid % 20, "position": pos, "price": 5.0,
                     "xp_next1": xp_, "p_start": 0.9, "p_play": 0.93, "ownership": own})
    pool = pd.DataFrame(rows)
    star = pool.index[pool.position == "FWD"][0]
    pool.loc[star, ["ownership", "xp_next1"]] = [60.0, 12.0]
    rivals = sample_rival_squads(pool, n_rivals=3000, rng=np.random.default_rng(0))
    rate = (rivals[:, star] > 0).mean()
    assert rate <= 0.62, f"started by {rate:.0%} of managers but owned by only 60%"
