import pandas as pd
from fpl.optimize.squad import Squad
from fpl.optimize.lineup import build_lineup, Lineup

XP = pd.DataFrame({
    "player_id": list(range(1, 16)),
    "web_name": [f"P{i}" for i in range(1, 16)],
    "team": ["T"] * 15,
    "position": (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3),
    "price": [5.0] * 15,
    "xp_next1": [6.0, 1.0,            # GKs: 1 starts, 2 benched
                 5.0, 4.9, 4.8, 4.7, 0.5,   # DEFs: last is weakest
                 9.0, 8.0, 7.0, 6.5, 0.4,   # MIDs
                 8.5, 7.5, 0.3],            # FWDs
    "xp_next5": [30.0] * 15,
    "p_start": [0.9] * 15,
    "e_minutes": [80.0] * 15,
    "confidence": ["high"] * 15,
    "flags": [[] for _ in range(15)],
})
SQUAD = Squad(
    player_ids=list(range(1, 16)),
    starting_ids=[1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
    total_cost=75.0,
    xp=0.0,
)


def test_returns_eleven_starters_and_four_bench():
    lu = build_lineup(SQUAD, XP)
    assert isinstance(lu, Lineup)
    assert len(lu.xi) == 11
    assert len(lu.bench) == 4


def test_bench_and_xi_partition_the_squad():
    lu = build_lineup(SQUAD, XP)
    assert set(lu.xi) | set(lu.bench) == set(SQUAD.player_ids)
    assert not set(lu.xi) & set(lu.bench)


def test_reserve_keeper_is_first_on_the_bench():
    lu = build_lineup(SQUAD, XP)
    assert lu.bench[0] == 2  # the non-starting GKP


def test_outfield_bench_ordered_by_descending_xp():
    lu = build_lineup(SQUAD, XP)
    outfield = lu.bench[1:]
    xp = XP.set_index("player_id")["xp_next1"]
    assert list(xp.loc[outfield]) == sorted(xp.loc[outfield], reverse=True)


def test_formation_string_matches_starters():
    lu = build_lineup(SQUAD, XP)
    assert lu.formation == "4-4-2"


def test_captain_is_highest_xp_starter():
    lu = build_lineup(SQUAD, XP)
    assert lu.captain == 8  # xp 9.0


def test_vice_is_second_highest_and_differs_from_captain():
    lu = build_lineup(SQUAD, XP)
    assert lu.vice == 13  # xp 8.5
    assert lu.vice != lu.captain


def test_lineup_xp_counts_captain_twice():
    lu = build_lineup(SQUAD, XP)
    xp = XP.set_index("player_id")["xp_next1"]
    expected = sum(xp.loc[lu.xi]) + xp.loc[lu.captain]
    assert abs(lu.xp - expected) < 1e-9
