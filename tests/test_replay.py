"""B13: a replay that carries real manager state, not a free weekly rebuild.

Each test below pins one rule the old harness did not obey. Where the live tool
already owns a rule -- free transfers, selling prices, chip legality, autosubs
-- the replay is required to reach it through the SAME function, so a rule can
never be right here and wrong in production.
"""
import pandas as pd
import pytest

from fpl.config import Config
from fpl.backtest.replay import (ManagerState, Decision, step, replay_season,
                                 compare_policies, hold_policy,
                                 expected_points_policy, oracle_rebuild_policy,
                                 selling_values, ORACLE_POLICIES)


def _pool(xp_next1=None):
    """A legal pool: 8 clubs, enough of each position to build two squads."""
    rows, pid = [], 1
    for t in range(8):
        for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for _ in range(n):
                rows.append({
                    "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                    "position": pos, "price": 5.0, "ownership": 5.0,
                    "xp_next1": 2.0, "xp_horizon": 10.0, "xp_next5": 10.0,
                    "p_play": 0.9, "p_start": 0.9, "e_minutes": 80.0,
                    "confidence": "high", "flags": [],
                })
                pid += 1
    df = pd.DataFrame(rows)
    if xp_next1 is not None:
        df["xp_next1"] = df["player_id"].map(xp_next1).fillna(df["xp_next1"])
    return df


POOL = _pool()
# 5 clubs x 3 players, respecting 2/5/5/3 and the club cap.
# Pool layout per club: GKP x2, DEF x5, MID x5, FWD x3, ids ascending.
CURRENT = [1, 2, 3, 18, 19, 20, 33, 38, 39, 53, 54, 55, 73, 74, 75]
# A LEGAL 4-4-2: one keeper, four defenders, four midfielders, two forwards.
# CURRENT[:11] is not legal -- it carries both keepers -- and autosub correctly
# refuses to touch an XI it cannot keep legal, which hid the substitution.
XI = [1, 3, 18, 19, 20, 38, 39, 53, 54, 73, 74]
BENCH = [2, 33, 55, 75]


def _actuals(points=1.0, minutes=90.0, ids=None):
    ids = ids if ids is not None else list(POOL.player_id)
    return pd.DataFrame({"player_id": ids, "round": 1,
                         "actual": [points] * len(ids),
                         "minutes": [minutes] * len(ids)})


def _cfg(**kw):
    return Config(budget=100.0, rank_sims=0, horizon_gw=1, max_paid_hits=2,
                  bench_floor_xp=0.0, **kw)


def _state(**kw):
    base = dict(squad=list(CURRENT), bank=5.0,
                purchase_prices={i: 5.0 for i in CURRENT}, free_transfers=1)
    base.update(kw)
    return ManagerState(**base)


def _fixed(squad, xi, chip=None):
    return lambda xp, state, gw, cfg: Decision(list(squad), list(xi), chip)


# --- rule 1: transfers are charged through the live free-transfer function ---

def test_a_transfer_within_the_free_allowance_costs_nothing():
    swap = [p for p in CURRENT if p != 1] + [4]
    result, after = step(POOL, _actuals(), _state(free_transfers=1), 1, _cfg(),
                         _fixed(swap, swap[:11]))
    assert result.transfers == 1
    assert result.hit_cost == 0
    assert after.free_transfers == 1      # spent one, accrued one


def test_a_transfer_beyond_the_allowance_is_charged_once():
    swap = [p for p in CURRENT if p not in (1, 2)] + [4, 5]
    result, after = step(POOL, _actuals(), _state(free_transfers=1), 1, _cfg(),
                         _fixed(swap, swap[:11]))
    assert result.transfers == 2
    assert result.hit_cost == 4
    assert after.hits == 4
    assert result.points == result.gross_points - 4


def test_unused_free_transfers_bank_up_to_the_cap():
    _, after = step(POOL, _actuals(), _state(free_transfers=4), 1, _cfg(),
                    _fixed(CURRENT, XI))
    assert after.free_transfers == 5
    _, again = step(POOL, _actuals(), _state(free_transfers=5), 1, _cfg(),
                    _fixed(CURRENT, XI))
    assert again.free_transfers == 5


# --- rule 2: a Wildcard or Free Hit makes the week's transfers free ---

def test_a_wildcard_week_pays_no_hit_however_many_moved():
    rebuilt = [p + 100 for p in range(1, 16)]
    rebuilt = list(POOL.player_id[:2]) + list(POOL.player_id[16:19]) \
        + list(POOL.player_id[31:34]) + list(POOL.player_id[46:48]) \
        + list(POOL.player_id[61:64]) + list(POOL.player_id[76:78])
    rebuilt = rebuilt[:15]
    result, after = step(POOL, _actuals(), _state(free_transfers=1), 2, _cfg(),
                         _fixed(rebuilt, rebuilt[:11], chip="wildcard"))
    assert result.transfers > 1
    assert result.hit_cost == 0
    # The chip consumes the gameweek's own free transfer, so the balance holds.
    assert after.free_transfers == 1
    assert after.chip_events == [{"chip": "wildcard", "event": 2}]


# --- rule 3: a Free Hit squad is discarded after its gameweek ---

def test_the_gameweek_after_a_free_hit_starts_from_the_permanent_squad():
    temporary = list(POOL.player_id[:2]) + list(POOL.player_id[16:21]) \
        + list(POOL.player_id[31:36]) + list(POOL.player_id[46:49])
    _, after = step(POOL, _actuals(), _state(bank=2.0), 2, _cfg(),
                    _fixed(temporary, temporary[:11], chip="freehit"))
    assert after.squad == temporary
    assert after.freehit_event == 2
    assert after.base_squad == CURRENT

    # The NEXT gameweek must plan from the fifteen the chip replaced, at the
    # bank held before it -- money the chip week freed up does not carry over.
    result, restored = step(POOL, _actuals(), after, 3, _cfg(),
                            hold_policy)
    assert set(restored.squad) == set(CURRENT)
    assert restored.bank == pytest.approx(2.0)
    assert restored.freehit_event is None


# --- rule 4: selling prices come from recorded purchase prices ---

def test_selling_value_takes_only_half_a_rise():
    pool = POOL.copy()
    pool.loc[pool.player_id == 1, "price"] = 5.4
    state = _state(purchase_prices={**{i: 5.0 for i in CURRENT}})
    assert selling_values(state, pool)[1] == pytest.approx(5.2)


def test_a_fall_is_taken_in_full():
    pool = POOL.copy()
    pool.loc[pool.player_id == 1, "price"] = 4.6
    assert selling_values(_state(), pool)[1] == pytest.approx(4.6)


# --- rule 5: chip legality goes through fpl.chips ---

def test_an_illegal_chip_is_not_played():
    """A Wildcard already used in this half cannot be played again, whatever
    the policy proposes."""
    state = _state(chip_events=[{"chip": "wildcard", "event": 4}])
    result, after = step(POOL, _actuals(), state, 6, _cfg(),
                         _fixed(CURRENT, XI, chip="wildcard"))
    assert result.chip is None
    assert after.chip_events == [{"chip": "wildcard", "event": 4}]


def test_a_second_half_chip_is_allowed():
    state = _state(chip_events=[{"chip": "wildcard", "event": 4}])
    result, _ = step(POOL, _actuals(), state, 25, _cfg(),
                     _fixed(CURRENT, XI, chip="wildcard"))
    assert result.chip == "wildcard"


# --- rule 6: scoring reaches realised_score, so autosubs and the armband apply ---

def test_a_blanking_starter_is_replaced_by_a_bench_player_who_played():
    actuals = _actuals()
    actuals.loc[actuals.player_id == 74, ["actual", "minutes"]] = [0.0, 0.0]
    result, _ = step(POOL, actuals, _state(), 1, _cfg(), _fixed(CURRENT, XI))
    assert 74 not in result.xi
    # FPL takes the FIRST bench player in order who played and keeps the shape
    # legal -- not the one in the same position. Here that is 33, a defender,
    # turning 4-4-2 into 5-4-1, which still satisfies the minimum per position.
    assert 33 in result.xi
    assert len(result.xi) == 11


def test_the_captain_is_paid_twice():
    result, _ = step(POOL, _actuals(points=3.0), _state(), 1, _cfg(),
                     _fixed(CURRENT, XI))
    # Eleven starters at 3 points, plus the armband paying one of them again.
    assert result.gross_points == pytest.approx(11 * 3.0 + 3.0)


def test_a_bench_boost_pays_the_bench_too():
    plain, _ = step(POOL, _actuals(points=3.0), _state(), 2, _cfg(),
                    _fixed(CURRENT, XI))
    boosted, _ = step(POOL, _actuals(points=3.0), _state(), 2, _cfg(),
                      _fixed(CURRENT, XI, chip="benchboost"))
    assert boosted.gross_points == pytest.approx(plain.gross_points + 4 * 3.0)


def test_a_triple_captain_pays_the_armband_a_third_time():
    plain, _ = step(POOL, _actuals(points=3.0), _state(), 2, _cfg(),
                    _fixed(CURRENT, XI))
    tripled, _ = step(POOL, _actuals(points=3.0), _state(), 2, _cfg(),
                      _fixed(CURRENT, XI, chip="triplecaptain"))
    assert tripled.gross_points == pytest.approx(plain.gross_points + 3.0)


# --- the season walk and the comparison ---

def test_state_carries_across_gameweeks():
    xp_by_gw = {1: POOL, 2: POOL, 3: POOL}
    actuals = pd.concat([_actuals().assign(round=gw) for gw in (1, 2, 3)])
    results, final = replay_season(xp_by_gw, actuals, _state(free_transfers=1),
                                   _cfg(), hold_policy)
    assert [r.gw for r in results] == [1, 2, 3]
    # Never transferred, so the balance banks to the cap and then stops.
    assert [r.free_transfers_after for r in results] == [2, 3, 4]
    assert final.points == pytest.approx(sum(r.points for r in results))


def test_the_oracle_is_marked_as_not_executable():
    """A free rebuild every gameweek ignores the squad, the bank, the free
    transfers and the chips. Reporting it beside real policies without saying
    so is what made the old harness misleading."""
    xp_by_gw = {1: POOL}
    actuals = _actuals()
    table = compare_policies(xp_by_gw, actuals, _state(), _cfg(),
                             policies={"hold": hold_policy,
                                       "oracle": oracle_rebuild_policy})
    assert set(table["policy"]) == {"hold", "oracle"}
    indexed = table.set_index("policy")
    assert bool(indexed.loc["oracle", "executable"]) is False
    assert bool(indexed.loc["hold", "executable"]) is True
    assert "oracle" in ORACLE_POLICIES


def test_the_comparison_reports_the_edge_against_the_field():
    xp_by_gw = {1: POOL}
    table = compare_policies(xp_by_gw, _actuals(points=2.0), _state(), _cfg(),
                             policies={"hold": hold_policy},
                             field_average={1: 10.0})
    row = table.iloc[0]
    assert row["mean_edge"] == pytest.approx(row["points"] - 10.0)


def test_the_expected_points_policy_spends_a_free_transfer_when_it_pays():
    """A strictly better player at the same price, one free transfer available."""
    upgrade = {p: 2.0 for p in POOL.player_id}
    upgrade[4] = 20.0                      # a GKP outside the current squad
    pool = _pool(xp_next1=upgrade)
    pool["xp_horizon"] = pool["xp_next1"]
    _, after = step(pool, _actuals(), _state(free_transfers=1), 1, _cfg(),
                    expected_points_policy)
    assert 4 in after.squad


def test_the_oracle_is_not_charged_for_its_free_rebuild():
    """Its premise is that the rules do not apply. Charging it hits made the
    'ceiling' score below doing nothing, which is not a ceiling."""
    rebuilt = list(POOL.player_id[:2]) + list(POOL.player_id[16:21]) \
        + list(POOL.player_id[31:36]) + list(POOL.player_id[46:49])
    result, _ = step(POOL, _actuals(), _state(free_transfers=1), 1, _cfg(),
                     lambda xp, st, gw, c: Decision(rebuilt, rebuilt[:11],
                                                    free_transfers=15))
    assert result.transfers > 1
    assert result.hit_cost == 0


def test_an_executable_policy_cannot_waive_its_own_hits():
    swap = [p for p in CURRENT if p not in (1, 2)] + [4, 5]
    result, _ = step(POOL, _actuals(), _state(free_transfers=1), 1, _cfg(),
                     _fixed(swap, swap[:11]))
    assert result.hit_cost == 4


# --- RB3: the replay scores the armband the tool would actually set ---

def test_the_replay_scores_the_risk_aware_captain_not_the_top_projection():
    """choose_captain values the armband as xp[c] + (1 - p_play[c]) * xp[vice],
    so a slightly lower projection with a good vice behind it can beat the raw
    top pick. realised_score used to reselect by raw xP alone, scoring a
    decision the tool never made."""
    pool = POOL.copy()
    frame = pool.set_index("player_id")
    # 73 (FWD): the raw top projection, certain to play. 38 (MID): a touch
    # lower, but a 20% chance of handing a strong vice the double.
    #   value(73) = 6.0 + 0.0 * 5.5 = 6.0
    #   value(38) = 5.5 + 0.2 * 6.0 = 6.7   -> the live armband goes to 38
    pool.loc[pool.player_id == 73, ["xp_next1", "p_play"]] = [6.0, 1.0]
    pool.loc[pool.player_id == 38, ["xp_next1", "p_play"]] = [5.5, 0.8]
    actuals = _actuals(points=2.0)

    result, _ = step(pool, actuals, _state(), 1, _cfg(), hold_policy)
    assert 73 in result.xi and 38 in result.xi
    assert result.captain == 38


def test_the_vice_takes_the_armband_when_the_captain_does_not_appear():
    actuals = _actuals(points=2.0)
    actuals.loc[actuals.player_id == 1, ["actual", "minutes"]] = [0.0, 0.0]
    actuals.loc[actuals.player_id == 3, "actual"] = 9.0
    decide = lambda xp, st, gw, c: Decision(list(CURRENT), list(XI), captain=1, vice=3)
    result, _ = step(POOL, actuals, _state(), 1, _cfg(), decide)
    assert result.captain == 3


def test_an_explicit_armband_is_honoured_over_the_top_projection():
    pool = POOL.copy()
    pool.loc[pool.player_id == 73, "xp_next1"] = 9.0     # would be the raw pick
    decide = lambda xp, st, gw, c: Decision(list(CURRENT), list(XI), captain=38, vice=39)
    result, _ = step(pool, _actuals(points=2.0), _state(), 1, _cfg(), decide)
    assert result.captain == 38


def test_the_expected_policy_sees_the_full_horizon():
    """With horizon columns present, a transfer worth little now and a lot
    later must be visible to the production policy."""
    pool = POOL.copy()
    for e in (1, 2, 3):
        pool[f"xp_gw{e}"] = pool["xp_next1"]
    # 4 is a GKP outside the squad: nothing this week, huge over the horizon.
    pool.loc[pool.player_id == 4, "xp_gw1"] = 0.5
    pool.loc[pool.player_id == 4, ["xp_gw2", "xp_gw3"]] = 30.0
    pool["xp_horizon"] = pool["xp_gw1"] + 0.85 * pool["xp_gw2"] + 0.85 ** 2 * pool["xp_gw3"]
    _, after = step(pool, _actuals(), _state(free_transfers=1), 1, _cfg(), expected_points_policy)
    assert 4 in after.squad
