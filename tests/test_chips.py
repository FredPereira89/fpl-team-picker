import pandas as pd
from fpl.optimize.lineup import Lineup
from fpl.optimize.chips import advise_chips, ChipAdvice, BENCH_BOOST_MIN_XP

SQUAD = list(range(1, 16))
TEAM_BY_PLAYER = {i: 1 for i in SQUAD}
LINEUP = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                formation="4-4-2", captain=1, vice=2, xp=60.0)
# Chip tests run from GW2, not GW1: Wildcard and Free Hit are illegal in an
# entry's opening gameweek, so a GW1 fixture cannot exercise either of them.
EVENT = 2


def _used(*names, event=1):
    """Chip uses as the DATED records advise_chips now takes."""
    return [{"chip": n, "event": event} for n in names]


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


def _counts(n=1, event=EVENT):
    return pd.DataFrame([{"team_id": 1, "event": event, "n_fixtures": n}])


def test_no_chip_recommended_in_a_normal_week():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [])
    assert isinstance(a, ChipAdvice)
    assert a.chip is None
    assert a.reason


def test_bench_boost_when_bench_is_strong():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT, [])
    assert a.chip == "benchboost"


def test_no_bench_boost_when_one_bench_player_is_weak():
    xp = _xp(bench_xp=BENCH_BOOST_MIN_XP + 1)
    xp.loc[xp.player_id == 15, "xp_next1"] = 0.1
    a = advise_chips(xp, LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [])
    assert a.chip != "benchboost"


def test_triple_captain_on_a_double_gameweek():
    a = advise_chips(_xp(captain_xp=12.0), LINEUP, SQUAD, _counts(n=2),
                     TEAM_BY_PLAYER, EVENT, [])
    assert a.chip == "triplecaptain"


def test_free_hit_when_several_players_blank():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(n=0), TEAM_BY_PLAYER, EVENT, [])
    assert a.chip == "freehit"


def test_wildcard_when_squad_riddled_with_problems():
    flags = [["Unavailable (i): injured"]] * 5 + [[] for _ in range(10)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(),
                     TEAM_BY_PLAYER, EVENT, [])
    assert a.chip == "wildcard"


def test_used_chips_are_never_suggested_again():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT, _used("benchboost"))
    assert a.chip != "benchboost"


def test_reason_always_explains_the_tradeoff():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT, [])
    assert len(a.reason) > 20


def test_fallback_is_honest_when_a_triggered_chip_is_already_used():
    """Bench qualifies for Bench Boost, but it's already been used -- the
    fallback reason must say so, not falsely claim the bench is weak.

    It names the gameweek and the window too: under the two-set rules "already
    used" is ambiguous, because a chip spent in the first half is back in the
    second."""
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT, _used("benchboost"))
    assert a.chip is None
    assert "was played in GW1" in a.reason
    assert "too weak" not in a.reason


# --- horizon-aware chip timing -------------------------------------------

def _multi_counts(rows):
    """rows: (team_id, event, n_fixtures)."""
    return pd.DataFrame(
        [{"team_id": t, "event": e, "n_fixtures": n} for t, e, n in rows]
    )


def test_squad_exposure_counts_doubles_and_blanks_per_event():
    """Two teams: team 1 doubles in GW5, team 2 blanks in GW5. The squad has
    10 players on team 1 and 5 on team 2, so GW5 is 10 doubles / 5 blanks."""
    from fpl.optimize.chips import squad_exposure

    team_by_player = {i: (1 if i <= 10 else 2) for i in SQUAD}
    counts = _multi_counts([
        (1, 4, 1), (2, 4, 1),
        (1, 5, 2), (2, 5, 0),
    ])

    exposure = squad_exposure(counts, SQUAD, team_by_player)

    assert exposure[4] == {"doubles": 0, "blanks": 0}
    assert exposure[5] == {"doubles": 10, "blanks": 5}


def _xp_with_events(per_event: dict[int, list[float]], flags=None):
    """per_event: {gameweek: [xp for each of the 15 squad players]}."""
    frame = _xp(flags=flags)
    for event, vals in per_event.items():
        frame[f"xp_gw{event}"] = vals
    first = min(per_event)
    frame["xp_next1"] = per_event[first]
    return frame


def test_captain_value_by_event_picks_the_best_squad_member_each_week():
    """The armband moves: player 1 is the pick in GW5, player 2 in GW6."""
    from fpl.optimize.chips import captain_value_by_event

    xp = _xp_with_events({
        5: [11.0, 5.0] + [1.0] * 13,
        6: [6.0, 14.0] + [1.0] * 13,
    })

    values = captain_value_by_event(xp, SQUAD)

    assert values[5] == 11.0
    assert values[6] == 14.0


def test_bench_value_by_event_sums_the_four_weakest_that_week():
    """Bench composition is not fixed -- it is whoever projects worst that
    gameweek, so the same squad has a 4.0 bench in GW5 and a 12.0 one in GW6."""
    from fpl.optimize.chips import bench_value_by_event

    xp = _xp_with_events({
        5: [10.0] * 11 + [1.0] * 4,
        6: [10.0] * 11 + [3.0] * 4,
    })

    values = bench_value_by_event(xp, SQUAD)

    assert values[5] == 4.0
    assert values[6] == 12.0


def test_patience_bar_rises_for_a_double_beyond_the_xp_horizon():
    """A GW20 double for the whole squad is invisible to the 5-GW projection,
    but it must still make the advisor harder to please in GW5."""
    from fpl.optimize.chips import patience_bar

    exposure = {e: {"doubles": 0, "blanks": 0} for e in range(5, 39)}
    exposure[20] = {"doubles": 15, "blanks": 0}

    bar, target = patience_bar(exposure, from_event=5, last_event=38,
                               horizon_last=9, kind="doubles", squad_size=15)

    assert bar > 1.0
    assert target == 20


def test_patience_fades_as_the_season_runs_out():
    """An unplayed chip scores zero, so the same full-squad double two weeks
    from the end must not hold the chip back as hard as one 30 weeks out."""
    from fpl.optimize.chips import patience_bar

    early_exposure = {20: {"doubles": 15, "blanks": 0}}
    late_exposure = {38: {"doubles": 15, "blanks": 0}}

    early, _ = patience_bar(early_exposure, from_event=5, last_event=38,
                            horizon_last=9, kind="doubles", squad_size=15)
    late, _ = patience_bar(late_exposure, from_event=36, last_event=38,
                           horizon_last=36, kind="doubles", squad_size=15)

    assert late < early
    assert late < 1.3


def test_triple_captain_is_held_for_a_better_week_inside_the_horizon():
    """GW5's captain clears the old 9.0 threshold outright, but GW7 projects a
    16.0 armband. Firing now would burn the chip on the worse of two weeks the
    model can actually see."""
    from fpl.optimize.chips import advise_chips

    xp = _xp_with_events({
        5: [10.0] + [4.0] * 10 + [1.0] * 4,
        6: [6.0] + [4.0] * 10 + [1.0] * 4,
        7: [6.0, 16.0] + [4.0] * 9 + [1.0] * 4,
    })
    counts = _multi_counts([(1, e, 1) for e in (5, 6, 7)])

    a = advise_chips(xp, LINEUP, SQUAD, counts, TEAM_BY_PLAYER, 5, [],
                     last_event=38)

    assert a.chip != "triplecaptain"
    assert a.hold_until == 7


def test_triple_captain_fires_when_this_week_is_the_best_visible_week():
    """The complement of the hold: if nothing ahead beats it, play it."""
    from fpl.optimize.chips import advise_chips

    xp = _xp_with_events({
        5: [16.0] + [4.0] * 10 + [1.0] * 4,
        6: [6.0] + [4.0] * 10 + [1.0] * 4,
        7: [6.0] + [4.0] * 10 + [1.0] * 4,
    })
    counts = _multi_counts([(1, e, 1) for e in (5, 6, 7)])

    a = advise_chips(xp, LINEUP, SQUAD, counts, TEAM_BY_PLAYER, 5, [],
                     last_event=38)

    assert a.chip == "triplecaptain"
    assert a.hold_until is None


def test_a_double_beyond_the_horizon_holds_a_merely_good_captain_week():
    """GW5's 10.0 armband clears the bare threshold, but a full-squad double in
    GW20 raises the bar above it. Raise the bar, don't veto."""
    from fpl.optimize.chips import advise_chips

    xp = _xp_with_events({
        5: [10.0] + [4.0] * 10 + [1.0] * 4,
        6: [6.0] + [4.0] * 10 + [1.0] * 4,
    })
    counts = _multi_counts([(1, 5, 1), (1, 6, 1), (1, 20, 2)])

    a = advise_chips(xp, LINEUP, SQUAD, counts, TEAM_BY_PLAYER, 5, [],
                     last_event=38)

    assert a.chip != "triplecaptain"
    assert a.hold_until == 20


def test_a_standout_captain_week_still_fires_through_a_beyond_horizon_double():
    """Same GW20 double, but a 25.0 armband this week clears even the raised
    bar -- structure may raise the threshold, never veto outright."""
    from fpl.optimize.chips import advise_chips

    xp = _xp_with_events({
        5: [25.0] + [4.0] * 10 + [1.0] * 4,
        6: [6.0] + [4.0] * 10 + [1.0] * 4,
    })
    counts = _multi_counts([(1, 5, 1), (1, 6, 1), (1, 20, 2)])

    a = advise_chips(xp, LINEUP, SQUAD, counts, TEAM_BY_PLAYER, 5, [],
                     last_event=38)

    assert a.chip == "triplecaptain"


def test_free_hit_is_held_for_a_worse_blank_week():
    """Five blanks now clears the Free Hit threshold, but GW20 blanks the whole
    squad. Spending the chip on the smaller problem wastes it."""
    from fpl.optimize.chips import advise_chips

    team_by_player = {i: (1 if i <= 10 else 2) for i in SQUAD}
    xp = _xp_with_events({
        5: [8.0] + [4.0] * 10 + [1.0] * 4,
        6: [7.0] + [4.0] * 10 + [1.0] * 4,
    })
    counts = _multi_counts([
        (1, 5, 1), (2, 5, 0),
        (1, 6, 1), (2, 6, 1),
        (1, 20, 0), (2, 20, 0),
    ])

    a = advise_chips(xp, LINEUP, SQUAD, counts, team_by_player, 5, [],
                     last_event=38)

    assert a.chip != "freehit"
    assert a.hold_until == 20


# --- Wildcard on squad QUALITY, not just squad health (2026-09-15) ---------
# The Wildcard fired only on a count of injury/doubt flags, so "my squad has
# drifted well below what my money could buy" -- the reason most managers
# actually play it -- could never trigger it. Measured on the real GW5 squad:
# an unconstrained rebuild projected 257.4 against 233.3 for holding, a gap of
# 24.1, of which free transfers could reach only 2.8.

from fpl.optimize.chips import SquadQuality


def test_wildcard_fires_when_a_rebuild_beats_what_transfers_can_reach():
    """Surplus clears the hits the same rebuild would otherwise cost."""
    q = SquadQuality(surplus=45.0, changes=10, hit_equivalent=40.0)
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [],
                     quality=q)
    assert a.chip == "wildcard"
    assert "45" in a.reason or "45.0" in a.reason


def test_wildcard_holds_when_the_rebuild_is_not_worth_the_hits():
    """The real GW5 case: a 21.4 surplus against 10 changes (40 points of
    hits) is not yet worth the chip."""
    q = SquadQuality(surplus=21.4, changes=10, hit_equivalent=40.0)
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [],
                     quality=q)
    assert a.chip != "wildcard"


def test_quality_alone_cannot_resurrect_a_spent_wildcard():
    q = SquadQuality(surplus=45.0, changes=10, hit_equivalent=40.0)
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT,
                     _used("wildcard"), quality=q)
    assert a.chip != "wildcard"
    assert "was played in GW1" in a.reason


def test_omitting_quality_leaves_the_advisor_exactly_as_it_was():
    """Mode 1 has no squad to compare against, and every existing caller
    passes nothing -- behaviour must be unchanged."""
    before = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [])
    after = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, EVENT, [],
                         quality=None)
    assert before.chip == after.chip and before.reason == after.reason


def test_the_injury_trigger_still_fires_without_any_quality_signal():
    """The two reasons are independent: four flagged players is a Wildcard
    whatever the rebuild says."""
    flags = [["Unavailable (i): knee"] for _ in range(4)] + [[] for _ in range(11)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER,
                     EVENT, [], quality=SquadQuality(surplus=0.0, changes=0,
                                                 hit_equivalent=0.0))
    assert a.chip == "wildcard"


# --- P0/B1: two sets of chips a season (2026-09-17 audit) ---

def test_a_first_half_wildcard_does_not_block_the_second_half_one():
    """Two Wildcards a season. The advisor used to see a single name and
    suppress the second for good."""
    flags = [["Unavailable (i): injured"]] * 5 + [[] for _ in range(10)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(event=25),
                     TEAM_BY_PLAYER, 25, _used("wildcard", event=5))
    assert a.chip == "wildcard"


def test_the_advisor_never_offers_a_chip_in_the_opening_gameweek():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(n=0, event=1),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip != "freehit"


def test_a_blocked_chip_is_explained_by_the_rule_that_blocked_it():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, EVENT,
                     _used("benchboost", event=1))
    assert a.chip is None
    assert "benchboost" in a.reason
