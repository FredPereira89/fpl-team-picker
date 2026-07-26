import pandas as pd
from fpl.data.cache import Cache
from fpl.data.archive import load_season_gws, verify_archive_integrity, ARCHIVE_URL
from fpl.backtest.aggregate import walk_forward_aggregate
from fpl.config import Config

CSV = (
    "name,position,team,element,GW,total_points,minutes,xP,"
    "expected_goals,expected_assists,bps,was_home,opponent_team\n"
    "Saka,MID,Arsenal,12,1,8,90,5.2,0.4,0.3,32,True,7\n"
    "Saka,MID,Arsenal,12,2,2,90,4.8,0.2,0.1,14,False,3\n"
    "Rice,MID,Arsenal,13,1,6,90,4.1,0.1,0.2,28,True,7\n"
)


class FakeResp:
    status_code = 200
    text = CSV

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return FakeResp()


def test_load_season_parses_csv(tmp_path):
    df = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    assert len(df) == 3
    assert set(df.columns) >= {"element", "GW", "total_points", "minutes", "xP"}
    assert df.loc[df.element == 12, "total_points"].sum() == 10


def test_load_season_uses_cache_on_second_call(tmp_path):
    cache, s = Cache(tmp_path), FakeSession()
    load_season_gws("2025-26", cache, session=s)
    load_season_gws("2025-26", cache, session=s)
    assert len(s.calls) == 1
    assert "2025-26" in s.calls[0]
    assert s.calls[0].startswith(ARCHIVE_URL.split("{")[0])


def test_integrity_check_passes_when_totals_match(tmp_path):
    gw = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    past = pd.DataFrame({"player_id": [12, 13], "season_name": ["2025/26"] * 2,
                         "total_points": [10, 6], "minutes": [180, 90]})
    result = verify_archive_integrity(gw, past, "2025/26")
    assert result["ok"] is True
    assert result["mismatched"] == 0


def test_integrity_check_flags_mismatch(tmp_path):
    gw = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    past = pd.DataFrame({"player_id": [12, 13], "season_name": ["2025/26"] * 2,
                         "total_points": [999, 6], "minutes": [180, 90]})
    result = verify_archive_integrity(gw, past, "2025/26")
    assert result["ok"] is False
    assert result["mismatched"] == 1


def test_walk_forward_reports_error_and_naive_comparison():
    past = pd.DataFrame({
        "player_id": [1, 1, 2, 2, 3, 3],
        "season_name": ["2024/25", "2025/26"] * 3,
        "total_points": [100, 110, 50, 45, 200, 190],
        "minutes": [3000, 3000, 2000, 2000, 3200, 3100],
    })
    out = walk_forward_aggregate(past, Config())
    assert out["n"] == 3
    assert out["mae"] >= 0
    assert "naive_mae" in out
    assert isinstance(out["beats_naive"], bool)


def test_walk_forward_handles_players_with_one_season():
    past = pd.DataFrame({
        "player_id": [1, 2],
        "season_name": ["2025/26", "2025/26"],
        "total_points": [100, 50],
        "minutes": [3000, 2000],
    })
    out = walk_forward_aggregate(past, Config())
    assert out["n"] == 0
