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


def test_rejects_not_yet_implemented_risk_profiles(tmp_path):
    for profile in ("template", "differential"):
        p = tmp_path / "config.yaml"
        p.write_text(f"risk: {{profile: {profile}}}\n")
        with pytest.raises(ValueError, match="not yet"):
            load_config(p)
