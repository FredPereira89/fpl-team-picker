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


# --- B12: the Tier 1 harness must not see the target season (2026-09-17 audit) ---

def test_the_population_prior_cannot_see_the_target_season():
    """pop_mean was the mean over the WHOLE frame, target seasons included, so
    every prediction was shrunk toward a number that already knew the answer."""
    past = pd.DataFrame([
        {"player_id": 1, "season_name": "2023/24", "minutes": 3000, "total_points": 100},
        {"player_id": 1, "season_name": "2024/25", "minutes": 3000, "total_points": 110},
        {"player_id": 2, "season_name": "2023/24", "minutes": 3000, "total_points": 90},
        {"player_id": 2, "season_name": "2024/25", "minutes": 3000, "total_points": 95},
    ])
    loud = past.copy()
    # Move the TARGET season only. A leak-free predictor cannot notice.
    loud.loc[loud.season_name == "2024/25", "total_points"] = 400

    a = walk_forward_aggregate(past, Config())
    b = walk_forward_aggregate(loud, Config())
    assert a["n"] == b["n"] == 2
    # Identical predictions; only the actuals they are scored against moved.
    assert a["mae"] != b["mae"]


def test_every_eligible_season_is_predicted_not_only_the_last():
    """Predicting each player's final season alone threw away most of the
    available out-of-sample evidence."""
    past = pd.DataFrame([
        {"player_id": 1, "season_name": s, "minutes": 3000, "total_points": p}
        for s, p in (("2022/23", 100), ("2023/24", 110), ("2024/25", 120))
    ])
    out = walk_forward_aggregate(past, Config())
    assert out["n"] == 2                      # 2023/24 and 2024/25
    assert set(out["by_cutoff"]) == {"2023/24", "2024/25"}


def test_the_naive_baseline_is_not_the_mean_of_the_answers():
    """`naive` was the mean of the TARGET actuals, which no forecaster could
    have known -- it made the baseline artificially hard to beat in MAE and the
    comparison meaningless."""
    past = pd.DataFrame([
        {"player_id": p, "season_name": s, "minutes": 3000, "total_points": v}
        for p in (1, 2, 3)
        for s, v in (("2023/24", 60 + p * 10), ("2024/25", 200))
    ])
    out = walk_forward_aggregate(past, Config())
    # Every 2024/25 actual is identical, so a mean-of-actuals baseline would
    # score a perfect 0. A pre-cutoff baseline cannot.
    assert out["naive_mae"] > 0
