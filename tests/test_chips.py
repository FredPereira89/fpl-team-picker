import pandas as pd
from fpl.optimize.lineup import Lineup
from fpl.optimize.chips import advise_chips, ChipAdvice, BENCH_BOOST_MIN_XP

SQUAD = list(range(1, 16))
TEAM_BY_PLAYER = {i: 1 for i in SQUAD}
LINEUP = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                formation="4-4-2", captain=1, vice=2, xp=60.0)


def _xp(bench_xp=1.0, captain_xp=8.0, flags=None):
    return pd.DataFrame({
        "player_id": SQUAD,
        "web_name": [f"P{i}" for i in SQUAD],
        "team": ["T1"] * 15,
        "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        "price": [5.0] * 15,
        "xp_next1": [captain_xp] + [4.0] * 10 + [bench_xp] * 4,
        "xp_next5": [30.0] * 15,
        "p_start": [0.9] * 15,
        "e_minutes": [80.0] * 15,
        "confidence": ["high"] * 15,
        "flags": flags or [[] for _ in SQUAD],
    })


def _counts(n=1):
    return pd.DataFrame([{"team_id": 1, "event": 1, "n_fixtures": n}])


def test_no_chip_recommended_in_a_normal_week():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, 1, [])
    assert isinstance(a, ChipAdvice)
    assert a.chip is None
    assert a.reason


def test_bench_boost_when_bench_is_strong():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, [])
    assert a.chip == "benchboost"


def test_no_bench_boost_when_one_bench_player_is_weak():
    xp = _xp(bench_xp=BENCH_BOOST_MIN_XP + 1)
    xp.loc[xp.player_id == 15, "xp_next1"] = 0.1
    a = advise_chips(xp, LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, 1, [])
    assert a.chip != "benchboost"


def test_triple_captain_on_a_double_gameweek():
    a = advise_chips(_xp(captain_xp=12.0), LINEUP, SQUAD, _counts(n=2),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip == "triplecaptain"


def test_free_hit_when_several_players_blank():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(n=0), TEAM_BY_PLAYER, 1, [])
    assert a.chip == "freehit"


def test_wildcard_when_squad_riddled_with_problems():
    flags = [["Unavailable (i): injured"]] * 5 + [[] for _ in range(10)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip == "wildcard"


def test_used_chips_are_never_suggested_again():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, ["benchboost"])
    assert a.chip != "benchboost"


def test_reason_always_explains_the_tradeoff():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, [])
    assert len(a.reason) > 20


def test_fallback_is_honest_when_a_triggered_chip_is_already_used():
    """Bench qualifies for Bench Boost, but it's already been used -- the
    fallback reason must say so, not falsely claim the bench is weak."""
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, ["benchboost"])
    assert a.chip is None
    assert "already used" in a.reason
    assert "too weak" not in a.reason
