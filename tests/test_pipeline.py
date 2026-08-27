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

    def element_summaries(self, player_ids, ttl_hours=None, progress=None):
        """Echo each element's totals back as a completed season.

        The BOOTSTRAP fixture above is written as a full season of history --
        which is what bootstrap-static actually shows pre-season. The pipeline now
        re-sources that baseline from element-summary history_past, so the fake
        client has to serve the same numbers there for the fixture to keep its
        meaning.
        """
        by_id = {e["id"]: e for e in BOOTSTRAP["elements"]}
        return {
            int(pid): {"history_past": [dict(by_id[int(pid)], season_name="2025/26")]}
            for pid in player_ids if int(pid) in by_id
        }


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
    project_root = Path(__file__).resolve().parent.parent
    src = ((project_root / "fpl" / "pipeline.py").read_text()
           + (project_root / "run_gameweek.py").read_text())
    for banned in ["mcp", "fantasy_pl", "fpl_mcp", "archive", "Skill"]:
        assert banned not in src, f"weekly path must not reference {banned!r}"


def test_stale_flag_propagates_to_report(tmp_path):
    class StaleClient(FakeClient):
        stale = True

    rec, _ = run(Config(), mode=1, from_event=1, root=tmp_path, client=StaleClient())
    assert rec.stale is True


def test_mode_two_bank_reflects_pre_transfer_squad_value(tmp_path):
    """Regression: bank must be recomputed from the CURRENT squad's value,
    not passed through unchanged -- otherwise a squad whose value changes
    after transfers reports a bank that doesn't reconcile against budget."""
    from fpl.data.normalize import normalize_players
    players = normalize_players(BOOTSTRAP)
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    current, club_count = [], {}
    cheap = players[players.player_id % 5 == 0].sort_values("player_id")
    for _, row in cheap.iterrows():
        pos, team_id, pid = row["position"], int(row["team_id"]), int(row["player_id"])
        if need.get(pos, 0) > 0 and club_count.get(team_id, 0) < 3:
            current.append(pid)
            need[pos] -= 1
            club_count[team_id] = club_count.get(team_id, 0) + 1
        if all(v == 0 for v in need.values()):
            break

    rec, xp = run(Config(max_paid_hits=2), mode=2, from_event=1, root=tmp_path,
                  client=FakeClient(), current_squad=current, bank=5.0, free_transfers=2)
    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    value_before = round(sum(prices[i] for i in current), 1)
    expected_bank = round(5.0 + value_before - rec.squad_value, 1)
    assert abs(rec.bank - expected_bank) < 1e-6
    # Fixture is chosen so the optimizer actually changes squad value --
    # otherwise the old (buggy) pass-through formula would coincidentally match.
    assert value_before != rec.squad_value


def test_mode_falls_back_honestly_when_no_current_squad_available(tmp_path):
    """Regression: requesting mode=2 with no current_squad (the CLI never
    fetches one today) must report BOTH mode and trust as reflecting the
    Mode 1 rebuild that actually ran, not the requested mode=2 label with
    the trust caveat silently dropped."""
    rec, _ = run(Config(), mode=2, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.mode == 1
    assert rec.trust != ""
    assert rec.transfers is None


def test_news_override_reaches_the_minutes_model(tmp_path):
    """A p_start override handed to run() must actually move that player's xP.

    minutes_model's own override handling is unit-tested, but nothing checked
    that the pipeline forwards `news` at all -- so a broken wire here would be
    invisible: the run still succeeds and just silently ignores the correction.
    """
    cfg = Config(budget=100.0)
    _, base = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    _, cut = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient(),
                 news={5: {"p_start_override": 0.0, "note": "benched",
                           "source": "test"}})

    def xp_of(frame, pid):
        return float(frame.loc[frame["player_id"] == pid].iloc[0]["xp_next5"])

    def p_start_of(frame, pid):
        return float(frame.loc[frame["player_id"] == pid].iloc[0]["p_start"])

    assert p_start_of(cut, 5) < p_start_of(base, 5)
    assert xp_of(cut, 5) < xp_of(base, 5)
    assert xp_of(cut, 6) == xp_of(base, 6), "other players must be untouched"


def test_clean_sheet_value_tracks_the_baseline_league_goal_rate(tmp_path):
    """Doubling every team's goals conceded in the baseline season must make
    clean sheets rarer, and so make defenders worth less.

    All teams move together, so the att/dfn ratios are unchanged and the only
    thing that can move is the league goal rate the pipeline estimates.
    """
    import copy
    leaky = copy.deepcopy(BOOTSTRAP)
    for e in leaky["elements"]:
        e["goals_conceded"] = e["goals_conceded"] * 2

    class LeakyClient(FakeClient):
        def bootstrap(self):
            return leaky

        def element_summaries(self, player_ids, ttl_hours=None, progress=None):
            by_id = {e["id"]: e for e in leaky["elements"]}
            return {
                int(pid): {"history_past": [dict(by_id[int(pid)], season_name="2025/26")]}
                for pid in player_ids if int(pid) in by_id
            }

    cfg = Config(budget=100.0, horizon_gw=3)
    _, base_xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    _, leaky_xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=LeakyClient())

    base_def = base_xp[base_xp.position == "DEF"].xp_next1.mean()
    leaky_def = leaky_xp[leaky_xp.position == "DEF"].xp_next1.mean()
    assert leaky_def < base_def
