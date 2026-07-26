import pandas as pd
from fpl.config import Config
from fpl.optimize.transfers import optimize_transfers, TransferPlan


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
