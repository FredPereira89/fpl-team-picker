from pathlib import Path
import pytest
from fpl.config import load_config, Config


def test_loads_defaults_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "budget: 100.0\nhorizon_gw: 5\n"
        "risk: {profile: balanced, ownership_weight: 0.0}\n"
        "news: {weight: 0.5, max_age_hours: 48}\n"
        "model: {form_half_life_gw: 3, form_max_weight: 0.6, shrinkage_minutes: 900}\n"
        "optimizer: {max_paid_hits: 2, hit_cost: 4, bench_weight: [0.15, 0.1, 0.05, 0.02]}\n"
        "odds: {provider: null}\n"
        "data: {cache_ttl_hours: 6, cache_ttl_matchday_hours: 1}\n"
        "entry_id: null\nfree_transfers: 1\n"
    )
    c = load_config(p)
    assert isinstance(c, Config)
    assert c.budget == 100.0
    assert c.risk_profile == "balanced"
    assert c.bench_weight == [0.15, 0.1, 0.05, 0.02]
    assert c.entry_id is None
    assert c.free_transfers == 1


def test_missing_keys_get_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("budget: 95.0\n")
    c = load_config(p)
    assert c.budget == 95.0
    assert c.horizon_gw == 5
    assert c.hit_cost == 4


def test_rejects_free_transfers_above_cap(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("free_transfers: 6\n")
    with pytest.raises(ValueError, match="free_transfers"):
        load_config(p)


def test_rejects_unknown_risk_profile(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("risk: {profile: reckless}\n")
    with pytest.raises(ValueError, match="risk.profile"):
        load_config(p)


def test_a_profile_listed_as_unimplemented_is_still_rejected(tmp_path, monkeypatch):
    """`template` and `differential` were on this list until they were wired
    into the objective on 2026-09-09, and the list is now empty. The guard it
    provided must survive: a profile the optimizer cannot honour has to fail at
    load rather than silently behave like `balanced`."""
    import fpl.config as config_module
    monkeypatch.setattr(config_module, "NOT_YET_IMPLEMENTED_PROFILES", {"template"})
    p = tmp_path / "config.yaml"
    p.write_text("risk: {profile: template}\n")
    with pytest.raises(ValueError, match="not yet"):
        load_config(p)


def test_setting_an_odds_provider_is_rejected_until_one_exists(tmp_path):
    """The field was loaded and never read: any value silently did nothing.
    Same treatment as the unimplemented risk profiles -- fail loudly instead."""
    p = tmp_path / "config.yaml"
    p.write_text("budget: 100.0\nodds: {provider: bet365}\n")
    with pytest.raises(ValueError, match="odds.provider"):
        load_config(p)


def test_differential_and_template_profiles_are_accepted_now_they_are_wired(tmp_path):
    """They were rejected because ownership_weight reached nothing. It now
    reaches the squad and transfer objectives, so the config must stop lying."""
    for profile in ("differential", "template"):
        p = tmp_path / f"{profile}.yaml"
        p.write_text(f"budget: 100\nrisk:\n  profile: {profile}\n  ownership_weight: 0.5\n")
        cfg = load_config(p)
        assert cfg.risk_profile == profile
        assert cfg.ownership_weight == 0.5


def test_ownership_weight_outside_zero_to_one_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("budget: 100\nrisk:\n  profile: differential\n  ownership_weight: 4\n")
    with pytest.raises(ValueError, match="ownership_weight"):
        load_config(p)


def test_rank_settings_load_with_sane_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("budget: 100\n")
    cfg = load_config(p)
    assert cfg.rank_sims > 0          # the distributional layer is on
    assert cfg.rank_candidates >= 1
    assert cfg.rank_target == 0.5     # "beat the median manager"


def test_rank_target_must_be_a_quantile(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("budget: 100\noptimizer:\n  rank_target: 1.4\n")
    with pytest.raises(ValueError, match="rank_target"):
        load_config(p)


def test_rank_can_be_switched_off(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("budget: 100\noptimizer:\n  rank_sims: 0\n")
    assert load_config(p).rank_sims == 0


def test_a_rank_target_too_extreme_to_simulate_is_rejected_at_load(tmp_path):
    """Failing at load beats failing twenty minutes into a run. Locating the
    99.999th percentile would need ten million rival managers."""
    p = tmp_path / "c.yaml"
    p.write_text("budget: 100\noptimizer:\n  rank_target: 0.99999\n")
    with pytest.raises(ValueError, match="cannot be resolved"):
        load_config(p)


def test_an_expensive_but_reachable_target_is_allowed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("budget: 100\noptimizer:\n  rank_target: 0.99\n")
    assert load_config(p).rank_target == 0.99


# --- R11: the bench floor is opt-in (2026-09-17 audit) ---

def test_the_bench_floor_is_off_unless_asked_for(tmp_path):
    """Every fresh build was silently solving a Bench Boost readiness problem:
    all four bench players forced over 2.5 xP, spending XI budget even when the
    chip was already spent -- and the setting was not in the checked-in config,
    so nobody could see it was on."""
    assert Config().bench_floor_xp == 0.0
    p = tmp_path / "config.yaml"
    p.write_text("optimizer:\n  bench_floor_xp: 2.5\n")
    assert load_config(p).bench_floor_xp == 2.5


def test_the_rank_layer_does_not_decide_transfers_by_default(tmp_path):
    """The rank objective scores ONE gameweek, so it cannot see the future gain
    a transfer is made for. Letting it pick the plan meant a move losing 0.2 now
    and gaining 8 over four weeks lost to a one-week swap, and a hit with strong
    payback was close to unselectable -- which defeats the point of charging
    four points for it."""
    assert Config().rank_transfers is False
    p = tmp_path / "config.yaml"
    p.write_text("optimizer:\n  rank_transfers: true\n")
    assert load_config(p).rank_transfers is True
