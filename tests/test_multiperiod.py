import pandas as pd
import pytest

from fpl.config import Config
from fpl.optimize.multiperiod import optimize_multi_period


POSITIONS = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def _pool():
    rows = []
    for group in range(2):
        for offset, position in enumerate(POSITIONS):
            pid = group * 15 + offset + 1
            rows.append({
                "player_id": pid,
                "web_name": f"P{pid}",
                "position": position,
                "team": f"T{group * 5 + offset % 5}",
                "price": 5.0,
                "xp_gw1": 2.0 if group == 0 else 1.0,
                "xp_gw2": 2.0 if group == 0 else 1.0,
                "xp_horizon": 4.0 if group == 0 else 2.0,
                "xp_next1": 2.0 if group == 0 else 1.0,
                "xp_next5": 4.0 if group == 0 else 2.0,
            })
    return pd.DataFrame(rows)


def _cfg(**changes):
    values = {
        "horizon_gw": 2,
        "horizon_decay": 1.0,
        "rank_sims": 0,
        "max_paid_hits": 1,
        "multi_period_ft_value": 1.5,
    }
    values.update(changes)
    return Config(**values)


def test_waits_for_a_future_fixture_instead_of_buying_it_early():
    pool = _pool()
    victim, target = 8, 23  # same position
    pool.loc[pool.player_id == victim, ["xp_gw1", "xp_gw2", "xp_horizon"]] = [10, 0, 10]
    pool.loc[pool.player_id == target, ["xp_gw1", "xp_gw2", "xp_horizon"]] = [0, 10, 10]

    plan = optimize_multi_period(pool, list(range(1, 16)), 0.0, 1, _cfg())

    assert plan.weeks[0].out_ids == []
    assert plan.weeks[0].free_transfers_after == 2
    assert plan.weeks[1].out_ids == [victim]
    assert plan.weeks[1].in_ids == [target]


def test_free_transfer_rollover_is_exact_and_capped_at_five():
    plan = optimize_multi_period(
        _pool(), list(range(1, 16)), 0.0, 4, _cfg(max_paid_hits=0))

    assert [w.free_transfers_before for w in plan.weeks] == [4, 5]
    assert [w.free_transfers_after for w in plan.weeks] == [5, 5]
    assert plan.terminal_free_transfers == 5
    assert all(w.paid_transfers == 0 for w in plan.weeks)


def test_a_second_immediate_upgrade_is_charged_as_a_hit():
    pool = _pool()
    victims, targets = [3, 8], [18, 23]  # DEF and MID
    for victim in victims:
        pool.loc[pool.player_id == victim,
                 ["xp_gw1", "xp_gw2", "xp_horizon"]] = [0, 0, 0]
    for target in targets:
        pool.loc[pool.player_id == target,
                 ["xp_gw1", "xp_gw2", "xp_horizon"]] = [20, 20, 40]

    plan = optimize_multi_period(pool, list(range(1, 16)), 0.0, 1, _cfg())

    assert set(plan.weeks[0].out_ids) == set(victims)
    assert set(plan.weeks[0].in_ids) == set(targets)
    assert plan.weeks[0].paid_transfers == 1
    assert plan.weeks[0].hit_cost == 4


def test_recorded_selling_value_controls_future_affordability():
    pool = _pool()
    victim, target = 8, 23
    pool.loc[pool.player_id == victim, "price"] = 6.0
    pool.loc[pool.player_id == target,
             ["price", "xp_gw1", "xp_gw2", "xp_horizon"]] = [6.0, 20, 20, 40]
    sale = {pid: float(pool.set_index("player_id").loc[pid, "price"])
            for pid in range(1, 16)}
    sale[victim] = 5.5

    short = optimize_multi_period(
        pool, list(range(1, 16)), 0.4, 1, _cfg(max_paid_hits=0), sale)
    enough = optimize_multi_period(
        pool, list(range(1, 16)), 0.5, 1, _cfg(max_paid_hits=0), sale)

    assert target not in short.weeks[0].in_ids
    assert enough.weeks[0].out_ids == [victim]
    assert enough.weeks[0].in_ids == [target]
    assert enough.weeks[0].bank_after == pytest.approx(0.0)


def test_every_week_is_a_legal_squad_and_lineup():
    plan = optimize_multi_period(_pool(), list(range(1, 16)), 0.0, 1, _cfg())
    frame = _pool().set_index("player_id")
    for week in plan.weeks:
        squad = frame.loc[week.squad_ids]
        xi = frame.loc[week.starting_ids]
        assert len(week.squad_ids) == 15
        assert len(week.starting_ids) == 11
        assert squad.position.value_counts().to_dict() == {
            "MID": 5, "DEF": 5, "FWD": 3, "GKP": 2}
        assert xi.position.value_counts().get("GKP", 0) == 1
        assert 3 <= xi.position.value_counts().get("DEF", 0) <= 5
        assert 2 <= xi.position.value_counts().get("MID", 0) <= 5
        assert 1 <= xi.position.value_counts().get("FWD", 0) <= 3
        assert squad.team.value_counts().max() <= 3


def test_ownership_tilt_never_recommends_a_raw_negative_transfer():
    """A differential tilt may pick a lower-raw-xP bench option internally,
    but the reported plan must never be a transfer that loses raw xP against
    simply holding -- see optimize_multi_period's docstring.
    """
    pool = _pool()
    victim, target = 8, 23  # same position (MID), both end up on the bench
    pool["ownership"] = 1.0
    # Every currently-owned player projects strongly (raw 10/gw) so the whole
    # group -- not just the victim -- is a live candidate for the bench; only
    # the victim's ownership singles it out for the tilt to punish.
    pool.loc[pool.player_id <= 15,
             ["xp_gw1", "xp_gw2", "xp_horizon", "xp_next1", "xp_next5"]] = [
                 10.0, 10.0, 20.0, 10.0, 20.0]
    pool.loc[pool.player_id == victim, "ownership"] = 100.0
    pool.loc[pool.player_id == target,
             ["xp_gw1", "xp_gw2", "xp_horizon", "xp_next1", "xp_next5", "ownership"]] = [
                 8.0, 8.0, 16.0, 8.0, 16.0, 0.0]

    cfg = _cfg(max_paid_hits=1, multi_period_ft_value=0.0,
               risk_profile="differential", ownership_weight=1.0)
    plan = optimize_multi_period(pool, list(range(1, 16)), 0.0, 1, cfg)

    assert plan.weeks[0].out_ids == []
    assert plan.weeks[0].in_ids == []
    assert plan.gain == 0.0
    assert plan.objective_xp == plan.baseline_objective_xp


def test_report_adapter_keeps_future_moves_explicitly_contingent():
    pool = _pool()
    pool.loc[pool.player_id == 8, ["xp_gw1", "xp_gw2", "xp_horizon"]] = [10, 0, 10]
    pool.loc[pool.player_id == 23, ["xp_gw1", "xp_gw2", "xp_horizon"]] = [0, 10, 10]
    plan = optimize_multi_period(pool, list(range(1, 16)), 0.0, 1, _cfg())

    first = plan.first_week_plan()
    assert first.strategy == "multi-period"
    assert first.n_transfers == 0
    assert first.future_plan[0]["event"] == 2
    assert first.terminal_free_transfers == plan.terminal_free_transfers
