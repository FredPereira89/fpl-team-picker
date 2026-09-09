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


# --- Ordered bench slots and a per-gameweek armband (2026-09-07 review) ---

def _bench_choice_pool():
    """A pool where the squad is forced except for three bench slots, and the
    budget leaves exactly £16.0m for them.

    Two affordable bench sets total the same 6 xP: one concentrated on the
    first substitute (6/0/0) and one spread flat (2/2/2). Averaging the four
    configured bench weights -- which both solvers did -- scores those two
    identically. Substitutions run in bench order, so they are not the same:
    Bench 1 comes on most weeks and Bench 3 almost never.
    """
    rows, pid = [], 1

    def add(pos, xp, price, team):
        nonlocal pid
        rows.append({"player_id": pid, "web_name": f"P{pid}", "team": team,
                     "position": pos, "price": price, "xp_next1": xp,
                     "xp_next5": xp, "p_start": 0.9, "e_minutes": 80.0,
                     "confidence": "high", "flags": []})
        pid += 1
        return pid - 1

    # Starting XI plus the reserve keeper: 12 players at 4.0 = 48.0.
    core = 0
    for pos, n in [("GKP", 1), ("DEF", 3), ("MID", 5), ("FWD", 2)]:
        for _ in range(n):
            add(pos, 10.0, 4.0, f"CORE{core // 3}")
            core += 1
    add("GKP", 0.5, 4.0, "GKX")
    # 8.0 + 4.0 + 4.0 = 16.0 exactly
    spike = [add("DEF", 6.0, 8.0, "SPIKE1"), add("DEF", 0.0, 4.0, "SPIKE2"),
             add("FWD", 0.0, 4.0, "SPIKE3")]
    # 5.3 x 3 = 15.9, also affordable; any mix of the two sets is not.
    flat = [add("DEF", 2.0, 5.3, "FLAT1"), add("DEF", 2.0, 5.3, "FLAT2"),
            add("FWD", 2.0, 5.3, "FLAT3")]
    return pd.DataFrame(rows), flat, spike


def test_bench_budget_goes_to_the_first_substitute_not_spread_flat():
    pool, flat, spike = _bench_choice_pool()
    s = optimize_squad(pool, Config(budget=64.0, bench_weight=[0.5, 0.1, 0.05, 0.02]),
                       xp_col="xp_next5")
    assert set(spike).issubset(set(s.player_ids))
    assert not set(flat) & set(s.player_ids)


def _horizon_pool():
    """Two candidates: a one-week spike and a steady accumulator."""
    rows, pid = [], 1
    for t in range(8):
        for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(n):
                rows.append({"player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                             "position": pos, "price": 4.0, "xp_next1": 1.0,
                             "xp_next5": 5.0, "xp_horizon": 5.0, "p_start": 0.9,
                             "e_minutes": 80.0, "confidence": "high", "flags": [],
                             **{f"xp_gw{e}": 1.0 for e in range(1, 6)}})
                pid += 1
    df = pd.DataFrame(rows)
    spike, steady = 5, 6  # both MID/DEF in the cheap pool, both affordable
    for col, val in [("xp_gw1", 15.0)] + [(f"xp_gw{e}", 0.0) for e in range(2, 6)]:
        df.loc[df.player_id == spike, col] = val
    for e in range(1, 6):
        df.loc[df.player_id == steady, f"xp_gw{e}"] = 6.0
    for pid_ in (spike, steady):
        total = sum(float(df.loc[df.player_id == pid_, f"xp_gw{e}"].iloc[0])
                    for e in range(1, 6))
        df.loc[df.player_id == pid_, ["xp_next5", "xp_horizon"]] = total
    return df, spike, steady


def test_the_armband_moves_to_whoever_is_best_that_gameweek():
    """A one-week spike should be captained that week and nobody's captain the
    rest of the horizon. A single armband over five gameweeks values it as if
    it had to be held all five, which is a decision that is free to revisit."""
    pool, spike, steady = _horizon_pool()
    s = optimize_squad(pool, Config(budget=100.0, horizon_decay=1.0),
                       xp_col="xp_horizon", must_include=[spike, steady])
    assert s.captains[1] == spike
    assert all(s.captains[e] == steady for e in range(2, 6))
    assert s.captain_id == spike  # this week's armband is what the report shows


def test_a_frame_without_per_gameweek_columns_still_picks_one_captain():
    """Callers outside the pipeline (and every test above) hand the solver a
    plain frame; it must fall back to a single captain rather than fail."""
    s = optimize_squad(POOL, CFG)
    assert s.captain_id in s.starting_ids
    assert list(s.captains) == [None]


# --- rank-relative scoring (2026-09-09 audit) ------------------------------

def test_effective_xp_is_unchanged_when_ownership_weight_is_zero():
    """The default must stay exactly the expected-points objective."""
    from fpl.optimize.objective import effective_xp
    from fpl.config import Config
    df = pd.DataFrame({"player_id": [1, 2], "xp_next1": [5.0, 5.0],
                       "ownership": [60.0, 1.0]})
    out = effective_xp(df, Config(ownership_weight=0.0), "xp_next1")
    assert list(out) == [5.0, 5.0]


def test_a_differential_profile_prefers_the_less_owned_of_two_equal_players():
    from fpl.optimize.objective import effective_xp
    from fpl.config import Config
    df = pd.DataFrame({"player_id": [1, 2], "xp_next1": [5.0, 5.0],
                       "ownership": [60.0, 1.0]})
    out = list(effective_xp(df, Config(risk_profile="differential",
                                       ownership_weight=0.5), "xp_next1"))
    assert out[1] > out[0]


def test_a_template_profile_prefers_the_widely_owned_of_two_equal_players():
    """Matching the field protects a rank you are defending; it is the opposite
    tilt, not the absence of one."""
    from fpl.optimize.objective import effective_xp
    from fpl.config import Config
    df = pd.DataFrame({"player_id": [1, 2], "xp_next1": [5.0, 5.0],
                       "ownership": [60.0, 1.0]})
    out = list(effective_xp(df, Config(risk_profile="template",
                                       ownership_weight=0.5), "xp_next1"))
    assert out[0] > out[1]


def test_ownership_tilt_never_outranks_a_real_points_gap():
    """A differential worth two points less must not be preferred at a sane
    weight -- the tilt is a tie-breaker on rank, not a licence to punt."""
    from fpl.optimize.objective import effective_xp
    from fpl.config import Config
    df = pd.DataFrame({"player_id": [1, 2], "xp_next1": [7.0, 5.0],
                       "ownership": [60.0, 0.5]})
    out = list(effective_xp(df, Config(risk_profile="differential",
                                       ownership_weight=0.5), "xp_next1"))
    assert out[0] > out[1]


def test_the_squad_solver_breaks_ties_toward_differentials_when_asked():
    """Two identical pools apart from ownership must not produce the same squad
    under a differential profile -- otherwise the setting is decorative."""
    from fpl.config import Config
    pool = POOL.copy()
    pool["ownership"] = [60.0 if i % 2 == 0 else 0.5 for i in range(len(pool))]
    balanced = optimize_squad(pool, Config(budget=100.0, horizon_gw=1))
    diff = optimize_squad(pool, Config(budget=100.0, horizon_gw=1,
                                       risk_profile="differential",
                                       ownership_weight=1.0))
    owned_balanced = pool.set_index("player_id").loc[balanced.player_ids, "ownership"].mean()
    owned_diff = pool.set_index("player_id").loc[diff.player_ids, "ownership"].mean()
    assert owned_diff < owned_balanced


def test_the_reported_squad_score_stays_honest_expected_points():
    """The tilt is a solver preference, not a points forecast. What the user is
    shown must remain real projected points, like xp_next5 beside xp_horizon."""
    from fpl.config import Config
    pool = POOL.copy()
    pool["ownership"] = 50.0
    cfg = Config(budget=100.0, horizon_gw=1, risk_profile="differential",
                 ownership_weight=1.0)
    squad = optimize_squad(pool, cfg)
    raw = pool.set_index("player_id").loc[squad.starting_ids, "xp_next5"].sum()
    assert squad.xp >= raw  # raw starters + the captain's doubled raw points
    assert squad.xp <= raw * 2


def test_reported_score_values_the_armband_across_the_whole_horizon():
    """The armband is re-chosen free every gameweek, so its value is the sum of
    each week's captain, DISCOUNTED like every other future point. Reporting one
    week's captain instead -- or the captain's undiscounted horizon total --
    misstates what the plan is worth."""
    from fpl.config import Config
    from fpl.optimize.objective import captain_values
    cfg = Config(budget=100.0, horizon_gw=3, horizon_decay=0.5)
    pool = POOL.copy()
    # Deliberately uneven weeks, so summing the parts differs from the total.
    pool["xp_gw1"] = pool["xp_next5"] * 0.6
    pool["xp_gw2"] = pool["xp_next5"] * 0.3
    pool["xp_gw3"] = pool["xp_next5"] * 0.1
    squad = optimize_squad(pool, cfg, xp_col="xp_next5")
    starters = pool.set_index("player_id").loc[squad.starting_ids, "xp_next5"].sum()
    armband = squad.xp - starters
    values = captain_values(pool, list(pool["player_id"]), cfg, "xp_next5")
    expected = sum(values[e][pid] for e, pid in squad.captains.items())
    assert armband == pytest.approx(expected, rel=1e-6)
    # And that is strictly less than handing over a whole undiscounted horizon.
    assert armband < pool.set_index("player_id").loc[squad.captain_id, "xp_next5"]


# --- candidate enumeration for rank selection (2026-09-09) -----------------
# A MILP maximises one linear objective. Rank is not linear in the squad, so
# it cannot be maximised directly -- the solver instead proposes its best few
# squads and the simulation picks between them.

def test_enumerate_returns_distinct_squads_best_first():
    from fpl.optimize.squad import enumerate_squads
    squads = enumerate_squads(POOL, Config(budget=100.0, horizon_gw=1), k=5)
    assert len(squads) == 5
    seen = {frozenset(s.player_ids) for s in squads}
    assert len(seen) == 5, "no-good cuts must exclude squads already returned"
    assert [round(s.xp, 6) for s in squads] == sorted(
        (round(s.xp, 6) for s in squads), reverse=True)


def test_the_first_enumerated_squad_is_the_plain_optimum():
    from fpl.optimize.squad import enumerate_squads
    cfg = Config(budget=100.0, horizon_gw=1)
    assert (set(enumerate_squads(POOL, cfg, k=3)[0].player_ids)
            == set(optimize_squad(POOL, cfg).player_ids))


def test_enumeration_stops_cleanly_when_the_pool_runs_out():
    """Exactly fifteen legal players across five clubs: one possible squad, so
    the second no-good cut makes the problem infeasible and enumeration has to
    stop rather than raise."""
    from fpl.optimize.squad import enumerate_squads
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    exact = pd.DataFrame([
        {"player_id": i, "web_name": f"X{i}", "team": f"C{i % 5}",
         "position": pos, "price": 4.0, "xp_next1": 2.0, "xp_next5": 10.0,
         "p_start": 0.9, "e_minutes": 80.0, "confidence": "high", "flags": []}
        for i, pos in enumerate(positions)
    ])
    squads = enumerate_squads(exact, Config(budget=200.0, horizon_gw=1), k=50)
    assert len(squads) == 1


def test_enumerated_squads_can_be_forced_genuinely_far_apart():
    """A no-good cut of one player produces near-identical squads, and on real
    data that made rank selection re-pick the plain optimum every time -- 7s of
    search that changed nothing. Candidates only inform the choice if they are
    actually different teams."""
    from fpl.optimize.squad import enumerate_squads
    cfg = Config(budget=100.0, horizon_gw=1)
    near = enumerate_squads(POOL, cfg, k=4, min_different=1)
    far = enumerate_squads(POOL, cfg, k=4, min_different=5)
    first = set(near[0].player_ids)
    assert len(first - set(near[1].player_ids)) == 1
    assert len(first - set(far[1].player_ids)) >= 5
