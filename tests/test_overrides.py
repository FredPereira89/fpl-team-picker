"""Team-news minutes overrides loaded from data/overrides.yaml.

`minutes_model` has always accepted a `news` dict of p_start overrides, but
nothing populated it -- corrections like "Arsenal's two first-choice centre-backs
are injured, so this player starts far more often than his own history implies"
had to be applied by hand in a scratch script. This makes them declarative,
reviewable and expiring.
"""
import pytest
from fpl.data.overrides import load_overrides

YAML = """
overrides:
  - player_id: 8
    name: Calafiori
    p_start_override: 0.80
    note: "Saliba and Timber both out"
    source: "FPL API status=i, checked 2026-08-26"
    until_gw: 5
  - player_id: 99
    name: Expired
    p_start_override: 0.10
    note: "stale note"
    source: "somewhere"
    until_gw: 1
"""


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "overrides.yaml"
    p.write_text(YAML, encoding="utf-8")
    return p


def test_loads_active_override_keyed_by_player_id(path):
    got = load_overrides(path, gw=2)
    assert got[8]["p_start_override"] == 0.80
    assert got[8]["note"] == "Saliba and Timber both out"
    assert got[8]["source"] == "FPL API status=i, checked 2026-08-26"


def test_drops_overrides_that_have_expired(path):
    """`until_gw` is inclusive -- an override for GW1 is gone by GW2."""
    assert 99 not in load_overrides(path, gw=2)
    assert 99 in load_overrides(path, gw=1)


def test_missing_file_is_not_an_error(tmp_path):
    assert load_overrides(tmp_path / "nope.yaml", gw=2) == {}


def test_empty_file_is_not_an_error(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_overrides(p, gw=2) == {}


def test_override_without_until_gw_never_expires(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, '
                 'note: n, source: s}\n', encoding="utf-8")
    assert 5 in load_overrides(p, gw=38)


def test_rejects_probability_outside_zero_to_one(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 1.4, '
                 'note: n, source: s}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="p_start_override"):
        load_overrides(p, gw=2)


def test_requires_a_source_so_overrides_stay_auditable(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, note: n}\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        load_overrides(p, gw=2)


# --- news.max_age_hours: stale team news is flagged, not silently trusted ---

def test_override_older_than_max_age_is_flagged_but_still_applied(tmp_path):
    """`until_gw` can keep an override alive for a month of gameweeks, which is
    far longer than an injury report stays true. Dropping it outright would be
    worse -- the model cannot make this correction for itself -- so it is
    applied and flagged."""
    from datetime import datetime, timezone
    p = tmp_path / "o.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, note: n, '
                 'source: s, checked_at: 2026-08-01}\n', encoding="utf-8")
    got = load_overrides(p, gw=2, max_age_hours=48,
                         now=datetime(2026, 9, 7, tzinfo=timezone.utc))
    assert got[5]["p_start_override"] == 0.9
    assert got[5]["stale"] is True
    assert got[5]["age_hours"] > 48


def test_freshly_checked_override_is_not_flagged(tmp_path):
    from datetime import datetime, timezone
    p = tmp_path / "o.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, note: n, '
                 'source: s, checked_at: 2026-09-07T08:00:00Z}\n', encoding="utf-8")
    got = load_overrides(p, gw=2, max_age_hours=48,
                         now=datetime(2026, 9, 7, 12, tzinfo=timezone.utc))
    assert got[5]["stale"] is False


def test_override_with_no_checked_at_cannot_be_verified_and_is_flagged(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, note: n, '
                 'source: s}\n', encoding="utf-8")
    got = load_overrides(p, gw=2, max_age_hours=48)
    assert got[5]["stale"] is True
    assert got[5]["age_hours"] is None


def test_without_a_max_age_nothing_is_flagged(tmp_path):
    """The check is opt-in: callers that pass no budget get the old behaviour."""
    p = tmp_path / "o.yaml"
    p.write_text('overrides:\n  - {player_id: 5, p_start_override: 0.9, note: n, '
                 'source: s}\n', encoding="utf-8")
    assert load_overrides(p, gw=2)[5]["stale"] is False


def test_the_repository_overrides_all_carry_a_checked_at_date():
    """A dated override is what makes the freshness check possible at all."""
    from pathlib import Path
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "data" /
                          "overrides.yaml").read_text(encoding="utf-8")) or {}
    for entry in raw.get("overrides") or []:
        assert entry.get("checked_at"), f"override {entry.get('player_id')} is undated"
