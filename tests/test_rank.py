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
    # 2x, not the 3x this asserted before rivals had to be LEGAL: in this
    # fixture every club is a single position and all forwards are
    # differentials, so a three-per-club XI can carry at most ten template
    # players and must start a differential up front. The pull toward the
    # template is unchanged; the arithmetic ceiling on it is not.
    assert picked_rate[owned > 10].mean() > 2 * picked_rate[owned < 10].mean()


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


# --- resolving extreme targets (2026-09-09 statistical review) -------------
# 400 rivals estimate the 99.99th percentile as the MAX of 400 draws: biased
# 12 points low with SD 5.6 against a true bar of 111. "Top of FPL" is the
# 99.99th percentile, so the setting that matters most was the broken one.

def test_required_rivals_scales_with_how_extreme_the_target_is():
    from fpl.optimize.rank import required_rivals, MIN_RIVALS
    assert required_rivals(0.5) == MIN_RIVALS
    assert required_rivals(0.99) >= 10_000
    assert required_rivals(0.999) >= 100_000
    assert required_rivals(0.9) < required_rivals(0.99) < required_rivals(0.999)


def test_a_target_too_extreme_to_resolve_is_refused_not_guessed():
    from fpl.optimize.rank import required_rivals, MAX_RIVALS
    with pytest.raises(ValueError, match="cannot be resolved"):
        required_rivals(0.999999)
    assert required_rivals(0.999) <= MAX_RIVALS


def test_the_field_bar_is_unbiased_at_an_extreme_target():
    """The whole point: the bar must be the real quantile, not the max of a
    small sample. Tested against a known distribution."""
    from fpl.optimize.rank import field_bar
    n_sims, truth = 200, 111.1
    pool = pd.DataFrame([
        {"player_id": i, "web_name": f"P{i}", "team": f"T{i % 20}", "team_id": i % 20,
         "position": ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i % 8],
         "price": 5.0, "xp_next1": 4.0, "p_start": 0.9, "p_play": 0.93,
         "ownership": 20.0} for i in range(80)])
    rng = np.random.default_rng(0)
    samples = rng.normal(55.0 / 12, 15.0 / np.sqrt(12), (80, n_sims))
    bar = field_bar(pool, samples, target=0.99, rng=rng)
    assert bar.shape == (n_sims,)
    # Against a 400-rival estimate the extreme bar is biased low; with enough
    # rivals it should sit near the field's own upper tail.
    small = field_bar(pool, samples, target=0.99, rng=rng, n_rivals=400)
    assert bar.mean() > small.mean()


def test_p_beat_bar_matches_p_beat_target_on_the_same_field():
    from fpl.optimize.rank import p_beat_bar
    rng = np.random.default_rng(0)
    mine = rng.normal(60, 12, 4000)
    rivals = rng.normal(50, 15, (RIVALS, 4000))
    bar = np.quantile(rivals, 0.9, axis=0)
    assert p_beat_bar(mine, bar) == pytest.approx(
        p_beat_target(mine, rivals, 0.9), abs=1e-9)


def test_candidate_scoring_accepts_a_precomputed_bar():
    """Extreme targets need a bar drawn from far more rivals than the dense
    rival-score matrix can hold, so it has to be passed in rather than
    recomputed from whatever small sample happens to be to hand."""
    from fpl.optimize.rank import score_candidate
    samples = _samples()
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=np.random.default_rng(1))
    rs = squad_scores(rivals, samples)
    squad = _fake_squad(IDS[:15], IDS[:11])
    high_bar = np.full(samples.shape[1], 1e6)
    out = score_candidate(squad, IDS, samples, rs, bar=high_bar)
    assert out["p_beat_target"] == 0.0        # nothing clears an impossible bar
    assert out["rank_percentile"] > 0.0       # still measured against real rivals


def test_the_captain_is_chosen_against_the_same_bar():
    from fpl.optimize.rank import best_captain_by_rank
    samples = _samples()
    rivals = sample_rival_squads(POOL, n_rivals=RIVALS, rng=np.random.default_rng(1))
    rs = squad_scores(rivals, samples)
    xi = IDS[:11]
    low = np.full(samples.shape[1], -1e6)
    assert best_captain_by_rank(xi, IDS, samples, rs, bar=low) in xi


# --- R2: generated rivals must be squads someone could actually own ---

def _crowded_pool():
    """Everyone the field loves is at one club and expensive."""
    import pandas as pd
    rows = []
    pid = 1
    for pos, n in (("GKP", 4), ("DEF", 12), ("MID", 12), ("FWD", 8)):
        for i in range(n):
            star = i < 5
            rows.append({"player_id": pid, "position": pos,
                         "team": "T1" if star else f"T{2 + i % 6}",
                         "price": 13.0 if star else 4.5,
                         "ownership": 80.0 if star else 3.0,
                         "xp_next1": 8.0 if star else 2.0, "p_play": 0.9})
            pid += 1
    return pd.DataFrame(rows)


def test_no_rival_starts_more_than_three_from_one_club():
    from fpl.optimize.rank import sample_rival_squads, MAX_PER_CLUB
    pool = _crowded_pool()
    rivals = sample_rival_squads(pool, n_rivals=300, rng=np.random.default_rng(0))
    club = pool["team"].to_numpy()
    for row in rivals:
        started = np.flatnonzero(row > 0)
        counts = {}
        for i in started:
            counts[club[i]] = counts.get(club[i], 0) + 1
        assert max(counts.values()) <= MAX_PER_CLUB


def _by_pos_sorted(pool):
    import numpy as np
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    return {p: np.flatnonzero(pos == p)[np.argsort(price[pos == p])]
            for p in ("GKP", "DEF", "MID", "FWD")}


def test_no_rival_xi_costs_more_than_a_legal_fifteen_allows():
    """Every sampled XI must be completable into a real fifteen within
    budget -- checked exactly via `_cheapest_legal_bench`, not the
    pool-wide "cheapest keeper + 3 cheapest outfielders" scalar a rival XI
    could pass while its own specific formation's true bench still blew
    the budget (C6-6). Uses `_formation_trap_pool`, not `_crowded_pool`:
    the latter puts every GKP on one club, which makes some XIs genuinely
    uncompletable regardless of price -- a club-cap pathology `_repair`
    already documents it may not resolve, not a price-ceiling bug."""
    from fpl.optimize.rank import sample_rival_squads, _cheapest_legal_bench
    pool = _formation_trap_pool()
    rivals = sample_rival_squads(pool, n_rivals=300, rng=np.random.default_rng(0),
                                 budget=100.0)
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    club = pool["team"].to_numpy()
    by_pos_sorted = _by_pos_sorted(pool)
    for row in rivals:
        xi = np.flatnonzero(row > 0)
        bench_cost = _cheapest_legal_bench(xi, pos, club, price, by_pos_sorted)
        assert bench_cost is not None
        assert price[xi].sum() + bench_cost <= 100.0 + 1e-9


def _formation_trap_pool():
    """A pool where the field's favoured 3-5-2 XI passes the old pool-wide
    ceiling but cannot actually be completed into a legal fifteen: the
    pool's 3 globally cheapest outfielders are MIDs, yet a 3-5-2 XI starts
    all 5 MIDs, so none of those cheap MIDs are available for its bench --
    which instead needs pricier DEF/FWD. Reproduces the exact scenario from
    the C6-6 audit finding."""
    import pandas as pd
    rows = []
    pid = 1
    for price in (4.0, 4.5, 5.0, 5.5):                        # 4 GKP
        rows.append({"player_id": pid, "position": "GKP", "team": f"T{pid % 8}",
                     "price": price, "ownership": 30.0, "xp_next1": 4.0, "p_play": 0.9})
        pid += 1
    for price in (6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4):    # 8 DEF, none cheap
        rows.append({"player_id": pid, "position": "DEF", "team": f"T{pid % 8}",
                     "price": price, "ownership": 30.0, "xp_next1": 5.0, "p_play": 0.9})
        pid += 1
    for price in (3.5, 3.7, 3.9, 8.0, 8.2, 8.4, 8.6, 8.8):    # 8 MID: 3 cheap + 5 pricier
        rows.append({"player_id": pid, "position": "MID", "team": f"T{pid % 8}",
                     "price": price, "ownership": 30.0,
                     "xp_next1": 6.0 if price > 5.0 else 2.0, "p_play": 0.9})
        pid += 1
    for price in (7.0, 7.2, 7.4, 7.6, 7.8):                   # 5 FWD, none cheap
        rows.append({"player_id": pid, "position": "FWD", "team": f"T{pid % 8}",
                     "price": price, "ownership": 30.0, "xp_next1": 6.0, "p_play": 0.9})
        pid += 1
    return pd.DataFrame(rows)


def test_a_ceiling_passing_xi_that_cannot_be_completed_is_repaired():
    """The exact scenario numerically reproduced during the C6-6 audit: the
    pool-wide ceiling (84.9) passes a 3-5-2 XI costing 78.8, but that XI's
    true cheapest legal bench costs 25.3, for a real total of 104.1 against
    a 100.0 budget -- genuinely infeasible despite passing the old check.
    `_repair` must not let such an XI survive."""
    from fpl.optimize.rank import _cheapest_legal_bench, _repair

    pool = _formation_trap_pool()
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    club = pool["team"].to_numpy()
    by_pos_sorted = _by_pos_sorted(pool)
    by_pos = {p: np.flatnonzero(pos == p) for p in ("GKP", "DEF", "MID", "FWD")}
    start_pull = np.ones(len(pool))

    gkp = [0]                                    # cheapest GKP
    def_ = [4, 5, 6]                              # 3 cheapest DEF
    mid = [15, 16, 17, 18, 19]                    # all 5 MID (the 5 pricier ones)
    fwd = [20, 21]                                # 2 cheapest FWD
    trap_xi = np.array(gkp + def_ + mid + fwd)

    trap_cost = price[trap_xi].sum()
    true_bench = _cheapest_legal_bench(trap_xi, pos, club, price, by_pos_sorted)
    assert trap_cost == pytest.approx(78.8)
    assert true_bench == pytest.approx(25.3)
    assert trap_cost + true_bench > 100.0          # confirms the trap is real

    repaired = _repair(trap_xi, pos, club, price, start_pull, by_pos,
                       by_pos_sorted, 100.0, np.random.default_rng(0))
    bench_cost = _cheapest_legal_bench(repaired, pos, club, price, by_pos_sorted)
    assert bench_cost is not None
    assert price[repaired].sum() + bench_cost <= 100.0 + 1e-9


def _club_coupling_pool():
    """Codex's re-review of C6-6: the XI already has two players from club
    A; the cheapest reserve GKP is also club A; but the ONLY remaining DEF
    in the whole pool is ALSO club A. Taking the cheap GKP from A first
    (2 in XI + 1 GKP = 3, still legal on its own) leaves no room for the
    mandatory DEF from A (would make 4, over MAX_PER_CLUB): a position-by-
    position greedy fill, cheapest first with no backtracking, commits to
    the GKP before it can see that DEF has nowhere else to go, and reports
    the whole bench impossible even though choosing the pricier, non-A GKP
    instead leaves it fully legal."""
    rows = [
        {"player_id": 1, "position": "GKP", "team": "C", "price": 5.0},   # XI
        {"player_id": 2, "position": "GKP", "team": "A", "price": 4.0},   # cheapest bench GKP
        {"player_id": 3, "position": "GKP", "team": "L", "price": 4.5},   # the legal alternative
        {"player_id": 4, "position": "DEF", "team": "D", "price": 6.0},   # XI
        {"player_id": 5, "position": "DEF", "team": "E", "price": 6.0},   # XI
        {"player_id": 6, "position": "DEF", "team": "F", "price": 6.0},   # XI
        {"player_id": 7, "position": "DEF", "team": "G", "price": 6.0},   # XI
        {"player_id": 8, "position": "DEF", "team": "A", "price": 6.0},   # the ONLY bench DEF
        {"player_id": 9, "position": "MID", "team": "A", "price": 7.0},   # XI, club A #1
        {"player_id": 10, "position": "MID", "team": "A", "price": 7.0},  # XI, club A #2
        {"player_id": 11, "position": "MID", "team": "H", "price": 7.0},  # XI
        {"player_id": 12, "position": "MID", "team": "I", "price": 7.0},  # XI
        {"player_id": 13, "position": "MID", "team": "J", "price": 7.0},  # XI
        {"player_id": 14, "position": "FWD", "team": "K", "price": 8.0}, # XI
        {"player_id": 15, "position": "FWD", "team": "M", "price": 3.0}, # bench
        {"player_id": 16, "position": "FWD", "team": "N", "price": 3.5}, # bench
    ]
    return pd.DataFrame(rows)


def test_cheapest_legal_bench_backtracks_past_a_blocking_first_choice():
    """A greedy, position-by-position fill (cheapest first, no
    backtracking) takes the club-A goalkeeper before it can see that the
    only available defender is ALSO club A, and wrongly reports the whole
    bench as impossible (the two together, plus the XI's own two from
    club A, would be four -- one over MAX_PER_CLUB). The legal, cheapest
    bench exists by using the pricier, non-A goalkeeper instead: 4.5 (GKP)
    + 6.0 (DEF) + 3.0 + 3.5 (FWD) = 17.0."""
    from fpl.optimize.rank import _cheapest_legal_bench

    pool = _club_coupling_pool()
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    club = pool["team"].to_numpy()
    by_pos_sorted = _by_pos_sorted(pool)

    xi = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]     # rows: 1 GKP, 4 DEF, 5 MID, 1 FWD
    bench_cost = _cheapest_legal_bench(xi, pos, club, price, by_pos_sorted)
    assert bench_cost == pytest.approx(17.0)


def test_cheapest_legal_bench_stays_fast_with_many_tied_prices():
    """Pruning that only compares cost spent so far against the best full
    solution provides no benefit once many candidates share a price: a
    partial path's cost cannot reach the full-solution bound until the
    LAST slot is filled, so every tied branch gets walked to full depth
    before the tie finally prunes it -- measured at over a second for a
    single call on a pool of this size before the fix (a tight lower bound
    on the cost of the slots still unfilled is needed to prune earlier).
    This pool mirrors the shape that reproduced it: ~180 players over 20
    clubs, every price identical."""
    import time
    from fpl.optimize.rank import _cheapest_legal_bench

    rows = []
    pid = 0
    for posn, n_per_club in (("GKP", 1), ("DEF", 3), ("MID", 3), ("FWD", 2)):
        for c in range(20):
            for _ in range(n_per_club):
                rows.append({"player_id": pid, "position": posn, "team": f"T{c}",
                            "price": 5.0})
                pid += 1
    pool = pd.DataFrame(rows)
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    club = pool["team"].to_numpy()
    by_pos_sorted = _by_pos_sorted(pool)

    # An XI (one of each position's first few rows) with no club-cap issue
    # of its own -- the tied prices, not a legality conflict, are what used
    # to blow up the search.
    xi = list(pool.index[pool.position == "GKP"][:1]) + \
        list(pool.index[pool.position == "DEF"][:4]) + \
        list(pool.index[pool.position == "MID"][:5]) + \
        list(pool.index[pool.position == "FWD"][:1])

    t0 = time.perf_counter()
    bench_cost = _cheapest_legal_bench(xi, pos, club, price, by_pos_sorted)
    dt = time.perf_counter() - t0

    assert bench_cost == pytest.approx(20.0)     # 4 bench slots at 5.0 each
    assert dt < 0.5, f"took {dt:.3f}s -- the tie-breaking bound has regressed"


def _two_club_pool(rng, n_clubs=6):
    """A club-concentrated pool, built the same way as the Codex re-review's
    own stress case: prices, ownership and projection drawn independently of
    each other and of club, with club skewed toward just two of `n_clubs` --
    which is what let two players end up each other's only legal same-
    position replacement below."""
    rows = []
    pid = 1
    for posn, n in (("GKP", 6), ("DEF", 20), ("MID", 20), ("FWD", 12)):
        for _ in range(n):
            club_ = (f"T{rng.integers(0, 2)}" if rng.random() < 0.6
                    else f"T{rng.integers(0, n_clubs)}")
            rows.append({"player_id": pid, "position": posn, "team": club_,
                        "price": round(rng.uniform(3.5, 14.0), 1),
                        "ownership": round(rng.uniform(0.5, 60.0), 1),
                        "xp_next1": round(rng.uniform(1.0, 9.0), 1), "p_play": 0.9})
            pid += 1
    return pd.DataFrame(rows)


def test_repair_escapes_a_two_player_deadlock():
    """Reproduced directly against the pre-fix `_repair`: this exact pool
    and starting XI put players 19 and 6 as each other's only legal same-
    position replacement (removing the pricier one always offers the other
    right back), and the picked set never changed at all across 60 traced
    passes -- a genuine 2-cycle, not merely a slow one, that no amount of
    extra REPAIR_PASSES could ever escape on its own. `_repair` must reach
    a fully legal state instead."""
    from fpl.optimize.rank import _repair, _cheapest_legal_bench, MAX_PER_CLUB

    pool = _two_club_pool(np.random.default_rng(0))
    price = pool["price"].to_numpy()
    pos = pool["position"].to_numpy()
    club = pool["team"].to_numpy()
    by_pos_sorted = _by_pos_sorted(pool)
    by_pos = {p: np.flatnonzero(pos == p) for p in ("GKP", "DEF", "MID", "FWD")}
    own = pool["ownership"].to_numpy()
    xp = pool["xp_next1"].to_numpy()
    start_pull = own * np.clip(xp, 0.0, None)

    picked = np.array([1, 6, 12, 16, 20, 21, 26, 30, 38, 54, 56])
    repaired = _repair(picked, pos, club, price, start_pull, by_pos, by_pos_sorted,
                       100.0, np.random.default_rng(999))

    counts: dict = {}
    for i in repaired:
        counts[club[i]] = counts.get(club[i], 0) + 1
    bench_cost = _cheapest_legal_bench(repaired, pos, club, price, by_pos_sorted)
    assert max(counts.values()) <= MAX_PER_CLUB
    assert bench_cost is not None
    assert price[repaired].sum() + bench_cost <= 100.0 + 1e-9


def test_repair_keeps_eleven_starters_and_one_captain():
    from fpl.optimize.rank import sample_rival_squads
    pool = _crowded_pool()
    rivals = sample_rival_squads(pool, n_rivals=100, rng=np.random.default_rng(0))
    for row in rivals:
        assert (row > 0).sum() == 11
        assert (row == 2).sum() == 1


def test_a_pool_without_club_or_price_columns_still_samples():
    from fpl.optimize.rank import sample_rival_squads
    rivals = sample_rival_squads(POOL.drop(columns=[c for c in ("team", "price")
                                                    if c in POOL.columns]),
                                 n_rivals=20, rng=np.random.default_rng(0))
    assert rivals.shape[0] == 20


# --- R1: ties on the bar are worth half ---

def test_landing_exactly_on_the_bar_is_worth_half():
    from fpl.optimize.rank import p_beat_target, p_beat_bar
    rivals = np.array([[10.0, 10.0, 10.0]] * 5)        # field median = 10 everywhere
    mine = np.array([10.0, 11.0, 9.0])
    assert p_beat_target(mine, rivals, 0.5) == pytest.approx((1 + 0.5 + 0) / 3)
    assert p_beat_bar(mine, np.array([10.0, 10.0, 10.0])) == pytest.approx(0.5)
