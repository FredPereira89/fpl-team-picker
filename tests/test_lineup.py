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


# --- The armband is worth the vice's points too (2026-09-07 review) ---

def _with_p_play(values, xp_overrides=None):
    """xp_next1 is UNCONDITIONAL, so a player's appearance odds and his
    projection have to move together for a fixture to mean anything."""
    df = XP.copy()
    df["p_play"] = [values.get(int(p), 1.0) for p in df["player_id"]]
    for pid, xp in (xp_overrides or {}).items():
        df.loc[df.player_id == pid, "xp_next1"] = xp
    return df


def test_a_doubtful_captain_is_worth_the_vices_re_roll():
    """If the captain does not appear at all, the vice's score doubles instead.
    A captain with a real chance of missing, backed by a strong vice, can be
    worth more than the top projection on its own -- which picking the top two
    outright cannot express."""
    # P8 is a 50/50 on an 18-point ceiling, so 9.0 unconditionally; P13 (8.5)
    # is nailed.
    #   C=P8:  9.0 + 0.5 * 8.5 = 13.25
    #   C=P13: 8.5 + 0.0 * 9.0 =  8.50
    lu = build_lineup(SQUAD, _with_p_play({8: 0.5}))
    assert lu.captain == 8
    assert lu.vice == 13


def test_the_re_roll_never_outweighs_a_real_projection():
    """A fringe starter is absent often enough that the vice would usually
    inherit the armband, but he brings nothing of his own. P11 at a 20% chance
    of a 2.0-point cameo is worth 0.4 unconditionally:
      C=P11: 0.4 + 0.8 * 9.0 = 7.6   vs   C=P8: 9.0 + 0.0 * 8.5 = 9.0
    """
    lu = build_lineup(SQUAD, _with_p_play({11: 0.2}, {11: 0.4}))
    assert lu.captain == 8


def test_a_player_who_cannot_appear_is_never_captain():
    lu = build_lineup(SQUAD, _with_p_play({8: 0.0}))
    assert lu.captain != 8


def test_uniform_appearance_odds_reproduce_the_old_ordering():
    """Where nobody carries extra risk this must not move the armband."""
    lu = build_lineup(SQUAD, _with_p_play({}))
    assert (lu.captain, lu.vice) == (8, 13)


# --- B8: the reported XI is the best legal eleven THIS week ---

def _squad_frame():
    import pandas as pd
    rows = []
    for pid, pos, now, later in [
        (1, "GKP", 3.0, 3.0), (2, "GKP", 1.0, 1.0),
        (3, "DEF", 4.0, 4.0), (4, "DEF", 4.0, 4.0), (5, "DEF", 4.0, 4.0),
        (6, "DEF", 1.0, 6.0),   # benched this week, a horizon starter
        (7, "DEF", 5.5, 2.0),   # the reverse: big this week, weak later
        (8, "MID", 4.0, 4.0), (9, "MID", 4.0, 4.0), (10, "MID", 4.0, 4.0),
        (11, "MID", 4.0, 4.0), (12, "MID", 0.5, 0.5),
        (13, "FWD", 4.0, 4.0), (14, "FWD", 4.0, 4.0), (15, "FWD", 0.5, 0.5),
    ]:
        rows.append({"player_id": pid, "position": pos, "xp_next1": now,
                     "xp_horizon": later, "p_play": 0.9})
    return pd.DataFrame(rows)


def test_best_xi_picks_the_best_legal_eleven_for_the_week():
    from fpl.optimize.lineup import best_xi
    xi = best_xi(list(range(1, 16)), _squad_frame(), xp_col="xp_next1")
    assert len(xi) == 11
    assert 7 in xi and 6 not in xi
    assert 1 in xi and 2 not in xi


def test_best_xi_respects_the_formation_limits():
    from fpl.optimize.lineup import best_xi
    frame = _squad_frame()
    # Make every defender enormous: at most five may start, and a forward
    # and two midfielders must still be in.
    frame.loc[frame.position == "DEF", "xp_next1"] = 20.0
    xi = best_xi(list(range(1, 16)), frame)
    pos = frame.set_index("player_id").loc[xi, "position"].value_counts()
    assert pos["GKP"] == 1 and pos["DEF"] == 5 and pos["MID"] >= 2 and pos["FWD"] >= 1


def test_build_lineup_re_picks_the_xi_on_this_weeks_projection():
    """The solver's XI serves the whole horizon; the report is for one week.
    A player who is more valuable as a horizon-long starter was being shown
    in this week's XI over one who is clearly better now."""
    from fpl.optimize.lineup import build_lineup
    from fpl.optimize.squad import Squad
    horizon_xi = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]      # 6 in, 7 out
    lineup = build_lineup(Squad(list(range(1, 16)), horizon_xi, 0.0, 0.0),
                          _squad_frame())
    assert 7 in lineup.xi and 6 in lineup.bench


def test_build_lineup_can_keep_the_supplied_xi():
    from fpl.optimize.lineup import build_lineup
    from fpl.optimize.squad import Squad
    horizon_xi = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    lineup = build_lineup(Squad(list(range(1, 16)), horizon_xi, 0.0, 0.0),
                          _squad_frame(), exact=False)
    assert set(lineup.xi) == set(horizon_xi)
