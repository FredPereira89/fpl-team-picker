from pathlib import Path
import pandas as pd
from fpl.config import Config
from fpl.pipeline import run
from fpl.report.weekly import Recommendation

BOOTSTRAP = {
    "teams": [
        {"id": t, "name": f"Team{t}", "short_name": f"T{t}",
         "strength_overall_home": 3, "strength_overall_away": 3}
        for t in range(1, 9)
    ],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ],
    "elements": [],
}
_pid = 1
for _t in range(1, 9):
    for _et, _n in [(1, 3), (2, 6), (3, 6), (4, 4)]:
        for _k in range(_n):
            BOOTSTRAP["elements"].append({
                "id": _pid, "web_name": f"P{_pid}", "team": _t, "element_type": _et,
                "now_cost": 45 + (_pid % 5) * 10, "status": "a", "news": "",
                "chance_of_playing_next_round": None, "minutes": 2500, "starts": 28,
                "total_points": 100 + _pid % 40, "goals_scored": _pid % 8,
                "assists": _pid % 5, "clean_sheets": 10, "goals_conceded": 30,
                "saves": 60 if _et == 1 else 0, "bonus": 10, "bps": 400,
                "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
                "expected_goals": str(_pid % 8), "expected_assists": str(_pid % 5),
                "expected_goals_conceded": "30.0", "selected_by_percent": "5.0",
                "defensive_contribution": 100 + _pid % 20,
            })
            _pid += 1

FIXTURES = []
_fid = 1
for _ev in range(1, 7):
    for _h, _a in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        FIXTURES.append({
            "id": _fid, "event": _ev, "team_h": _h, "team_a": _a,
            "team_h_difficulty": 3, "team_a_difficulty": 3,
            "kickoff_time": f"2026-08-2{_ev}T19:00:00Z", "finished": False,
        })
        _fid += 1


class FakeClient:
    stale = False

    def bootstrap(self):
        return BOOTSTRAP

    def fixtures(self):
        return FIXTURES


def test_mode_one_produces_a_valid_recommendation(tmp_path):
    rec, xp = run(Config(budget=100.0), mode=1, from_event=1, root=tmp_path,
                  client=FakeClient())
    assert isinstance(rec, Recommendation)
    assert len(rec.squad_ids) == 15
    assert len(rec.lineup.xi) == 11
    assert len(rec.lineup.bench) == 4
    assert rec.lineup.captain in rec.lineup.xi


def test_mode_one_respects_budget(tmp_path):
    rec, _ = run(Config(budget=100.0), mode=1, from_event=1, root=tmp_path,
                 client=FakeClient())
    assert rec.squad_value <= 100.0 + 1e-6


def test_returns_contract_frame(tmp_path):
    from fpl.model.xp import CONTRACT_COLUMNS
    _, xp = run(Config(), mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert list(xp.columns) == CONTRACT_COLUMNS


def test_pipeline_imports_no_mcp_or_archive():
    """The headless path must not depend on MCP, skills, or the backtest archive."""
    src = Path("fpl/pipeline.py").read_text() + Path("run_gameweek.py").read_text()
    for banned in ["mcp", "fantasy_pl", "fpl_mcp", "archive", "Skill"]:
        assert banned not in src, f"weekly path must not reference {banned!r}"


def test_stale_flag_propagates_to_report(tmp_path):
    class StaleClient(FakeClient):
        stale = True

    rec, _ = run(Config(), mode=1, from_event=1, root=tmp_path, client=StaleClient())
    assert rec.stale is True
