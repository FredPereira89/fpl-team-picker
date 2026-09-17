import pandas as pd
import pytest

from fpl.config import Config
from fpl.optimize.actions import (ChipAction, wildcard_action, freehit_action,
                                  bench_boost_value, triple_captain_value)
from fpl.optimize.lineup import Lineup


def _pool():
    """A pool big enough to build two legal squads from, with a clear optimum."""
    rows = []
    pid = 1
    for pos, n in (("GKP", 6), ("DEF", 15), ("MID", 15), ("FWD", 9)):
        for i in range(n):
            rows.append({
                "player_id": pid,
                "web_name": f"{pos}{i}",
                "team": f"T{pid % 10}",
                "position": pos,
                "price": 4.0 + (i % 5) * 0.5,
                "ownership": 5.0,
                "xp_next1": 1.0 + (i % 7) * 0.4,
                "xp_next5": 5.0 + (i % 7) * 2.0,
                "xp_horizon": 4.5 + (i % 7) * 1.8,
                "p_play": 0.9,
            })
            pid += 1
    return pd.DataFrame(rows)


def _cfg():
    return Config(budget=100.0, rank_sims=0, bench_floor_xp=0.0)


def _current(pool):
    """A deliberately poor legal 15: the cheapest, lowest-projected of each."""
    picks = []
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        sub = pool[pool.position == pos].nsmallest(n, "xp_horizon")
        picks += [int(i) for i in sub.player_id]
    return picks


def _selling(pool, current):
    frame = pool.set_index("player_id")
    return {i: float(frame.loc[i, "price"]) for i in current}


def test_a_wildcard_returns_the_rebuild_it_measured():
    """The advisor computed a rebuild only to get a scalar, threw the squad
    away, and returned the limited-transfer squad instead -- so confirming a
    Wildcard applied a team the Wildcard had never chosen."""
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    action = wildcard_action(pool, cfg, current, bank=5.0,
                             selling=_selling(pool, current), xp_col="xp_horizon")
    assert isinstance(action, ChipAction)
    assert action.chip == "wildcard"
    assert len(action.squad_ids) == 15
    assert len(action.starting_ids) == 11
    assert action.temporary is False
    assert action.value > 0
    # And it is a real rebuild, not the squad it started from.
    assert set(action.squad_ids) != set(current)


def test_a_wildcard_rebuild_never_charges_a_points_hit():
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    action = wildcard_action(pool, cfg, current, bank=5.0,
                             selling=_selling(pool, current), xp_col="xp_horizon")
    assert action.transfers.hit_cost == 0


def test_a_free_hit_squad_is_optimised_for_one_week_only():
    """A Free Hit lasts a gameweek, so it is chosen on xp_next1. Choosing it on
    the horizon buys players for weeks the squad will not exist in."""
    pool, cfg = _pool(), _cfg()
    current = _current(pool)
    selling = _selling(pool, current)
    action = freehit_action(pool, cfg, current, bank=5.0, selling=selling)
    assert action.chip == "freehit"
    assert action.temporary is True
    assert len(action.squad_ids) == 15
    frame = pool.set_index("player_id")
    one_week = float(frame.loc[action.starting_ids, "xp_next1"].sum())
    horizon_pick = wildcard_action(pool, cfg, current, bank=5.0, selling=selling,
                                   xp_col="xp_horizon")
    assert one_week >= float(frame.loc[horizon_pick.starting_ids, "xp_next1"].sum())


def test_bench_boost_is_worth_the_real_bench_not_the_four_lowest():
    """The advisor approximated the bench as the four numerically lowest
    projections in the squad, which is not necessarily a legal bench."""
    pool = _pool()
    lineup = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                    formation="4-4-2", captain=1, vice=2, xp=40.0)
    frame = pool.set_index("player_id")
    expected = float(frame.loc[[12, 13, 14, 15], "xp_next1"].sum())
    assert bench_boost_value(lineup, pool) == pytest.approx(expected)


def test_triple_captain_is_worth_one_more_captain_return():
    pool = _pool()
    lineup = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                    formation="4-4-2", captain=3, vice=2, xp=40.0)
    frame = pool.set_index("player_id")
    captain_xp = float(frame.loc[3, "xp_next1"])
    p_play = float(frame.loc[3, "p_play"])
    vice_xp = float(frame.loc[2, "xp_next1"])
    expected = p_play * captain_xp + (1 - p_play) * vice_xp
    assert triple_captain_value(lineup, pool) == pytest.approx(expected)
