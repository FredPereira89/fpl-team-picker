import pandas as pd
import pytest
from fpl.config import Config
from fpl.optimize.squad import optimize_squad, Squad

CFG = Config(budget=100.0)


def make_pool(n_per_team=8, n_teams=10):
    """A pool rich enough that a valid 15 always exists."""
    rows, pid = [], 1
    for t in range(n_teams):
        for i in range(n_per_team):
            pos = ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i % 8]
            rows.append({
                "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                "position": pos, "price": 4.0 + (i % 5) * 1.5,
                "xp_next1": 1.0 + (pid % 7) * 0.4, "xp_next5": 5.0 + (pid % 7) * 1.3,
                "p_start": 0.9, "e_minutes": 80.0, "confidence": "high", "flags": [],
            })
            pid += 1
    return pd.DataFrame(rows)


POOL = make_pool()


def test_returns_exactly_fifteen_players():
    s = optimize_squad(POOL, CFG)
    assert isinstance(s, Squad)
    assert len(s.player_ids) == 15
    assert len(set(s.player_ids)) == 15


def test_position_split_is_two_five_five_three():
    s = optimize_squad(POOL, CFG)
    picked = POOL[POOL.player_id.isin(s.player_ids)]
    counts = picked.position.value_counts().to_dict()
    assert counts["GKP"] == 2 and counts["DEF"] == 5
    assert counts["MID"] == 5 and counts["FWD"] == 3


def test_budget_never_exceeded():
    s = optimize_squad(POOL, CFG)
    assert s.total_cost <= CFG.budget + 1e-6


def test_max_three_players_per_club():
    s = optimize_squad(POOL, CFG)
    picked = POOL[POOL.player_id.isin(s.player_ids)]
    assert picked.team.value_counts().max() <= 3


def test_starting_eleven_is_valid_formation():
    s = optimize_squad(POOL, CFG)
    xi = POOL[POOL.player_id.isin(s.starting_ids)]
    assert len(s.starting_ids) == 11
    c = xi.position.value_counts().to_dict()
    assert c.get("GKP", 0) == 1
    assert 3 <= c.get("DEF", 0) <= 5
    assert 2 <= c.get("MID", 0) <= 5
    assert 1 <= c.get("FWD", 0) <= 3


def test_starters_are_a_subset_of_the_squad():
    s = optimize_squad(POOL, CFG)
    assert set(s.starting_ids).issubset(set(s.player_ids))


def test_tighter_budget_still_produces_valid_squad():
    s = optimize_squad(POOL, Config(budget=80.0))
    assert len(s.player_ids) == 15
    assert s.total_cost <= 80.0 + 1e-6


def test_known_optimum_on_small_pool():
    """Two clear tiers: the optimiser must take every premium it can afford."""
    rows = []
    pid = 1
    for t in range(6):
        for pos, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(count):
                premium = pid % 4 == 0
                rows.append({
                    "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                    "position": pos, "price": 4.0, "xp_next1": 1.0,
                    "xp_next5": 50.0 if premium else 1.0, "p_start": 0.9,
                    "e_minutes": 80.0, "confidence": "high", "flags": [],
                })
                pid += 1
    pool = pd.DataFrame(rows)
    s = optimize_squad(pool, Config(budget=100.0))
    picked = pool[pool.player_id.isin(s.player_ids)]
    # every pick is affordable at 4.0 x 15 = 60 <= 100, so it maximises premiums
    # subject to 3-per-club and the 2/5/5/3 split
    assert (picked.xp_next5 == 50.0).sum() == 15


def test_banned_players_are_excluded():
    banned = list(POOL.player_id[:5])
    s = optimize_squad(POOL, CFG, banned=banned)
    assert not set(s.player_ids) & set(banned)


def test_must_include_players_are_selected():
    forced = [int(POOL.player_id.iloc[0]), int(POOL.player_id.iloc[4])]
    s = optimize_squad(POOL, CFG, must_include=forced)
    assert set(forced).issubset(set(s.player_ids))


def test_infeasible_budget_raises():
    pricey = POOL.copy()
    pricey["price"] = 20.0
    with pytest.raises(ValueError, match="infeasible"):
        optimize_squad(pricey, Config(budget=50.0))


def test_optimises_on_requested_horizon_column():
    short = optimize_squad(POOL, CFG, xp_col="xp_next1")
    long = optimize_squad(POOL, CFG, xp_col="xp_next5")
    assert len(short.player_ids) == len(long.player_ids) == 15


# --- P5a: captaincy in the objective (2026-08-27 audit) ---

def test_reported_xp_counts_the_captain_twice():
    """Captaincy is roughly a sixth of a gameweek score. Leaving it out of the
    objective meant the solver had no reason to prefer a high ceiling."""
    s = optimize_squad(POOL, CFG)
    starters = POOL[POOL.player_id.isin(s.starting_ids)]
    assert s.xp == pytest.approx(starters.xp_next5.sum() + starters.xp_next5.max(), abs=1e-3)


def test_captain_is_a_starter():
    s = optimize_squad(POOL, CFG)
    assert s.captain_id in s.starting_ids


def test_captain_is_the_best_starter_available():
    s = optimize_squad(POOL, CFG)
    starters = POOL[POOL.player_id.isin(s.starting_ids)].set_index("player_id")
    assert starters.loc[s.captain_id, "xp_next5"] == starters.xp_next5.max()
