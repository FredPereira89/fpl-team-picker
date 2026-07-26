import pandas as pd
from fpl.optimize.lineup import Lineup
from fpl.optimize.chips import ChipAdvice
from fpl.optimize.transfers import TransferPlan
from fpl.report.weekly import render, Recommendation

XP = pd.DataFrame({
    "player_id": list(range(1, 16)),
    "web_name": [f"Player{i}" for i in range(1, 16)],
    "team": ["Arsenal"] * 15,
    "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
    "price": [5.0] * 15,
    "xp_next1": [4.0] * 15,
    "xp_next5": [20.0] * 15,
    "p_start": [0.9] * 15,
    "e_minutes": [80.0] * 15,
    "confidence": ["high"] * 13 + ["low"] * 2,
    "flags": [[] for _ in range(13)] + [["Doubtful: 25% chance of playing"],
                                        ["Limited data: no Premier League minutes"]],
})
LINEUP = Lineup(xi=[1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14], bench=[2, 7, 12, 15],
                formation="4-4-2", captain=8, vice=13, xp=52.0)


def _rec(**kw):
    base = dict(
        gw=1, deadline="Fri 21 Aug 2026, 18:30 UK", mode=1, lineup=LINEUP,
        squad_ids=list(range(1, 16)), transfers=None,
        chip=ChipAdvice(None, "No chip recommended this week — bench too weak."),
        bank=0.5, squad_value=99.5, flags=[], stale=False,
        trust="Model beats both baselines in every position.",
    )
    base.update(kw)
    return Recommendation(**base)


def test_contains_all_required_sections():
    out = render(_rec(), XP)
    for section in ["## Gameweek 1", "### Starting XI", "### Bench (in order)",
                    "### Captain", "### Transfers this week", "### Chip watch",
                    "### Budget", "### Flags"]:
        assert section in out


def test_formation_and_deadline_in_header():
    out = render(_rec(), XP)
    assert "4-4-2" in out
    assert "Fri 21 Aug 2026, 18:30 UK" in out


def test_bench_is_numbered_in_order():
    out = render(_rec(), XP)
    bench_block = out.split("### Bench (in order)")[1].split("###")[0]
    for n in range(1, 5):
        assert f"{n}." in bench_block


def test_captain_and_vice_named_with_reasoning():
    out = render(_rec(), XP)
    block = out.split("### Captain")[1].split("###")[0]
    assert "Player8" in block and "Player13" in block
    assert len(block.strip().splitlines()) >= 2  # names plus a why line


def test_no_transfer_message_when_none_recommended():
    out = render(_rec(), XP)
    assert "No transfer recommended" in out


def test_transfer_rendered_with_net_gain():
    plan = TransferPlan(out_ids=[3], in_ids=[4], n_transfers=1, hit_cost=0,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=60.0, baseline_xp=57.5, gain=2.5)
    out = render(_rec(transfers=plan), XP)
    assert "Player3" in out and "Player4" in out
    assert "2.5" in out


def test_never_uses_past_tense_action_verbs():
    plan = TransferPlan(out_ids=[3], in_ids=[4], n_transfers=1, hit_cost=4,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=56.0, baseline_xp=55.0, gain=1.0)
    out = render(_rec(transfers=plan), XP).lower()
    for banned in ["transferred", "captained", "benched him", "i have made"]:
        assert banned not in out


def test_uses_recommendation_language():
    out = render(_rec(), XP).lower()
    assert "recommend" in out or "suggest" in out


def test_flags_on_squad_players_are_surfaced():
    out = render(_rec(), XP)
    assert "Doubtful: 25% chance of playing" in out
    assert "Limited data" in out


def test_stale_data_warning_shown():
    out = render(_rec(stale=True), XP)
    assert "stale" in out.lower()


def test_trust_summary_included():
    out = render(_rec(trust="LOW CONFIDENCE — DEF failed validation"), XP)
    assert "LOW CONFIDENCE" in out


def test_hit_cost_disclosed_when_taken():
    plan = TransferPlan(out_ids=[3, 5], in_ids=[4, 6], n_transfers=2, hit_cost=4,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=56.0, baseline_xp=55.0, gain=1.0)
    out = render(_rec(transfers=plan), XP)
    assert "-4" in out or "−4" in out
