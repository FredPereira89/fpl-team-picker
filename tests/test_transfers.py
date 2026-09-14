import pandas as pd
import pytest
from fpl.config import Config
from fpl.optimize.transfers import optimize_transfers, selling_price, TransferPlan


def make_pool():
    rows, pid = [], 1
    for t in range(8):
        for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(n):
                rows.append({
                    "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                    "position": pos, "price": 5.0, "xp_next1": 2.0,
                    "xp_next5": 10.0, "p_start": 0.9, "e_minutes": 80.0,
                    "confidence": "high", "flags": [],
                })
                pid += 1
    return pd.DataFrame(rows)


POOL = make_pool()
# BUG FIX 1: the brief's original CURRENT construction
# (POOL[position==X].player_id[:N] per position, concatenated) put all 15
# players on team T0, since the pool is built team-by-team and team0 alone
# supplies the full quota of every position. That violates the 3-per-club
# cap by construction, making the n=0 baseline (and every other search
# depth) infeasible -- every test failed with "max() iterable argument is
# empty" under the brief's original fixture. This explicit list spans
# exactly 5 clubs (T0-T4), 3 players each, respecting the 2/5/5/3 position
# split -- verified against the actual solver before dispatch.
CURRENT = [1, 2, 3, 18, 19, 20, 33, 38, 39, 53, 54, 55, 73, 74, 75]


def _pool_with_star(star_xp=60.0):
    pool = POOL.copy()
    # BUG FIX 2: the brief's original target selection ("first non-current
    # MID in raw pool order") picked a player from T0 -- but T0 is already
    # at the 3-per-club cap in CURRENT, so bringing that player in forces a
    # SECOND transfer just to free up a T0 slot, breaking the "clear
    # upgrade = 1 transfer" test. Picking a target from a club with ZERO
    # current-squad members avoids the cap collision, so a clean 1-for-1
    # swap is genuinely achievable and optimal -- verified against the
    # actual solver before dispatch.
    current_clubs = set(POOL[POOL.player_id.isin(CURRENT)].team)
    target = pool[(~pool.player_id.isin(CURRENT)) & (pool.position == "MID")
                  & (~pool.team.isin(current_clubs))].iloc[0]
    pool.loc[pool.player_id == target.player_id, "xp_next5"] = star_xp
    return pool, int(target.player_id)


def test_baseline_is_always_reported_first():
    cfg = Config(max_paid_hits=1)
    best, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert options[0].n_transfers == 0
    assert options[0].hit_cost == 0
    assert isinstance(best, TransferPlan)


def test_no_transfer_when_squad_already_optimal():
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert best.n_transfers == 0
    assert best.out_ids == [] and best.in_ids == []


def test_takes_a_free_transfer_for_a_clear_upgrade():
    pool, star = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert best.n_transfers == 1
    assert star in best.in_ids


def test_hit_cost_applied_beyond_free_allowance():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=2, hit_cost=4)
    _, options = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    two = next(o for o in options if o.n_transfers == 2)
    assert two.hit_cost == 4
    assert abs(two.net_xp - (two.gross_xp - 4)) < 1e-6


def test_search_depth_scales_with_banked_free_transfers():
    cfg = Config(max_paid_hits=1)
    _, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=5, cfg=cfg)
    assert max(o.n_transfers for o in options) == 6


def test_free_transfers_incur_no_hit():
    cfg = Config(max_paid_hits=0)
    _, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=3, cfg=cfg)
    assert all(o.hit_cost == 0 for o in options)


def test_gain_is_relative_to_baseline():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, options = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert abs(best.gain - (best.net_xp - options[0].net_xp)) < 1e-6
    assert best.gain >= 0


def test_result_squad_respects_all_constraints():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    picked = pool[pool.player_id.isin(best.squad_ids)]
    assert len(best.squad_ids) == 15
    assert picked.team.value_counts().max() <= 3
    counts = picked.position.value_counts().to_dict()
    assert counts["GKP"] == 2 and counts["DEF"] == 5
    assert counts["MID"] == 5 and counts["FWD"] == 3


def test_transfer_counts_are_balanced():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert len(best.out_ids) == len(best.in_ids) == best.n_transfers


def test_budget_respects_bank_plus_sale_proceeds():
    pool = POOL.copy()
    pool["price"] = 5.0
    expensive = pool[~pool.player_id.isin(CURRENT)].iloc[0].player_id
    pool.loc[pool.player_id == expensive, ["price", "xp_next5"]] = [9.0, 99.0]
    cfg = Config(max_paid_hits=0, budget=75.0)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.5, free_transfers=1, cfg=cfg)
    cost = pool[pool.player_id.isin(best.squad_ids)]["price"].sum()
    assert cost <= 15 * 5.0 + 0.5 + 1e-6


# --- P6: FPL selling price (2026-08-27 audit) ---

def test_selling_price_returns_half_the_rise_rounded_down():
    assert selling_price(purchase=5.0, now=5.3) == 5.1   # 0.3 rise -> keep 0.1
    assert selling_price(purchase=5.0, now=5.2) == 5.1
    assert selling_price(purchase=5.0, now=5.1) == 5.0   # a single 0.1 rise is not bankable
    assert selling_price(purchase=5.0, now=5.4) == 5.2


def test_selling_price_takes_the_full_loss_when_a_player_falls():
    assert selling_price(purchase=5.0, now=4.7) == 4.7


def test_selling_price_of_an_untouched_player_is_its_price():
    assert selling_price(purchase=5.0, now=5.0) == 5.0


def test_budget_uses_selling_value_not_market_value():
    """A squad that has risen 0.5 cannot be sold for 0.5 — half the rise is
    lost, so a transfer affordable at market value may not be affordable at all.
    """
    pool, star = _pool_with_star()
    pool.loc[pool.player_id == star, "price"] = 6.0
    cfg = Config(max_paid_hits=0)

    at_market, _ = optimize_transfers(pool, CURRENT, bank=1.0, free_transfers=1, cfg=cfg)
    assert star in at_market.squad_ids

    # every current player bought at 4.5, now worth 5.0 -> sells for 4.7
    selling = {pid: 4.7 for pid in CURRENT}
    at_selling, _ = optimize_transfers(pool, CURRENT, bank=1.0, free_transfers=1, cfg=cfg,
                                       selling_prices=selling)
    assert star not in at_selling.squad_ids


def test_holding_the_squad_stays_feasible_at_selling_prices():
    """Selling value is only charged on the way out — keeping a player who has
    risen must never look unaffordable."""
    cfg = Config(max_paid_hits=0)
    selling = {pid: 4.7 for pid in CURRENT}
    best, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=1, cfg=cfg,
                                       selling_prices=selling)
    assert options[0].n_transfers == 0


# --- rank-aware transfer choice (2026-09-09) -------------------------------
# The distributional layer was wired into the squad build and NOT into the
# weekly transfer run, which is the one anyone actually uses. So the objective
# it exists to replace -- expected points -- was still deciding every week.

def test_enumerate_transfer_plans_returns_genuinely_different_plans():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cur = list(CURRENT)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(budget=100.0, horizon_gw=5), xp_col="xp_next5", k=4,
                                     min_different=1)
    assert len(plans) >= 2
    seen = {frozenset(p.squad_ids) for p in plans}
    assert len(seen) == len(plans)


def test_every_enumerated_plan_is_a_legal_fifteen():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cur = list(CURRENT)
    pos = POOL.set_index("player_id")["position"]
    for plan in enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                         cfg=Config(budget=100.0, horizon_gw=5), xp_col="xp_next5", k=4):
        counts = pos.loc[plan.squad_ids].value_counts()
        assert len(plan.squad_ids) == 15
        assert counts.get("GKP", 0) == 2 and counts.get("DEF", 0) == 5
        assert counts.get("MID", 0) == 5 and counts.get("FWD", 0) == 3


def test_plans_carry_the_hit_they_would_cost():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cur = list(CURRENT)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=0,
                                     cfg=Config(budget=100.0, horizon_gw=5), xp_col="xp_next5", k=5)
    for plan in plans:
        assert plan.hit_cost == max(0, plan.n_transfers - 0) * Config().hit_cost


def test_rank_scoring_charges_a_plan_for_its_hit():
    """A transfer costing -4 has to beat the field by MORE than one that does
    not. Scoring the squads without the penalty silently makes hits free."""
    import numpy as np
    from fpl.optimize.rank import score_candidate
    from fpl.optimize.squad import Squad
    ids = [int(i) for i in POOL["player_id"]]
    rng = np.random.default_rng(0)
    samples = rng.poisson(3.0, (len(ids), 2000)).astype(float)
    rivals = rng.normal(35.0, 8.0, (200, 2000))
    squad = Squad(ids[:15], ids[:11], 100.0, 0.0)
    free = score_candidate(squad, ids, samples, rivals)
    hit = score_candidate(squad, ids, samples, rivals, penalty=4.0)
    assert hit["p_beat_target"] < free["p_beat_target"]
    assert hit["mean_points"] == pytest.approx(free["mean_points"] - 4.0)


# The weekly report quotes `plan.gain` as "suggested net gain of N xP across
# the horizon". Only optimize_transfers ever filled it in; enumerate_transfer_plans
# left it at the dataclass default, so once the rank layer was wired into mode 2
# (which enumerates rather than optimizes) EVERY weekly run reported a gain of
# exactly 0.0 -- a real +2.9 xP transfer read as buying nothing. Found on the
# real GW5 2026/27 run, 2026-09-14.

def test_enumerated_plans_report_their_gain_over_holding():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cur = list(CURRENT)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(budget=100.0, horizon_gw=5),
                                     xp_col="xp_next5", k=4)
    hold = [p for p in plans if p.n_transfers == 0]
    assert hold, "the 0-transfer baseline should always be among the candidates"
    baseline = hold[0].net_xp
    assert all(p.baseline_xp == baseline for p in plans)
    for p in plans:
        assert p.gain == round(p.net_xp - baseline, 3)


def test_the_hold_plan_reports_no_gain_over_itself():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cur = list(CURRENT)
    plans = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1,
                                     cfg=Config(budget=100.0, horizon_gw=5),
                                     xp_col="xp_next5", k=4)
    hold = [p for p in plans if p.n_transfers == 0][0]
    assert hold.gain == 0.0


def test_a_real_upgrade_is_not_reported_as_zero_gain():
    """The regression that mattered: a plan that genuinely improves the squad
    must report the improvement, not the dataclass default."""
    from fpl.optimize.transfers import enumerate_transfer_plans
    pool, _star = _pool_with_star()
    plans = enumerate_transfer_plans(pool, list(CURRENT), bank=5.0, free_transfers=1,
                                     cfg=Config(budget=100.0, horizon_gw=5),
                                     xp_col="xp_next5", k=4)
    best = max(plans, key=lambda p: p.net_xp)
    assert best.n_transfers > 0
    assert best.gain > 0.0
    hold = [p for p in plans if p.n_transfers == 0][0]
    assert best.gain == round(best.net_xp - hold.net_xp, 3)


# --- the bench floor belongs to squad BUILDS, not weekly transfers ---------
# With one free transfer a bench floor is either infeasible or forces a bad
# move: a bench cannot be repaired one player at a time. It applies to the
# wildcard rebuild, which really does restructure all fifteen.

def test_weekly_transfer_plans_ignore_the_bench_floor():
    from fpl.optimize.transfers import enumerate_transfer_plans
    cfg = Config(budget=100.0, horizon_gw=5, bench_floor_xp=2.5)
    plain = Config(budget=100.0, horizon_gw=5, bench_floor_xp=0.0)
    cur = list(CURRENT)
    a = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1, cfg=cfg,
                                 xp_col="xp_next5", k=4)
    b = enumerate_transfer_plans(POOL, cur, bank=5.0, free_transfers=1, cfg=plain,
                                 xp_col="xp_next5", k=4)
    assert [set(p.squad_ids) for p in a] == [set(p.squad_ids) for p in b]


def test_a_rebuild_can_be_asked_to_respect_the_bench_floor():
    """The wildcard path opts in explicitly, so the surplus it reports is
    measured against a squad the builder would actually produce."""
    from fpl.optimize.transfers import _solve, _budget_and_cost
    # POOL's xp_next1 is flat, so the floor would have nothing to bind against:
    # give a third of the pool a non-playing week, as bench fodder really has.
    pool = POOL.copy()
    weak = pool.index % 3 == 0
    pool.loc[weak, "xp_next1"] = 0.3
    cfg = Config(budget=100.0, horizon_gw=5, bench_floor_xp=1.5)
    cur = set(CURRENT)
    budget, cost = _budget_and_cost(pool, cur, 5.0, None)
    solved = _solve(pool, cur, budget, 15, cfg, "xp_next5", cost=cost,
                    bench_floor=1.5)
    assert solved is not None
    chosen, starters, _ = solved
    v = pool.set_index("player_id")["xp_next1"]
    benched = [i for i in chosen if i not in starters]
    assert benched, "a 15 always has a bench"
    assert all(float(v.loc[i]) >= 1.5 for i in benched)


def test_the_floor_is_judged_on_one_gameweek_not_the_horizon():
    """Zetterer projected 0.74 for the week and 2.58 over the horizon: judged on
    the horizon he clears a 2.5 floor while still never playing."""
    from fpl.optimize.transfers import _solve, _budget_and_cost
    pool = POOL.copy()
    pool["xp_next1"] = 0.5      # nobody plays this week
    pool["xp_next5"] = 40.0     # but everyone looks fine over the horizon
    cfg = Config(budget=100.0, horizon_gw=5, bench_floor_xp=2.5)
    cur = set(CURRENT)
    budget, cost = _budget_and_cost(pool, cur, 5.0, None)
    solved = _solve(pool, cur, budget, 15, cfg, "xp_next5", cost=cost,
                    bench_floor=2.5)
    assert solved is None, "the floor must read the gameweek, not the horizon"
