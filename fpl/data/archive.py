"""Historical per-GW data (backtest only).

NEVER import this from the weekly pipeline. It exists solely as a cold-start
crutch for pre-season validation; from GW2 the project's own cached
element-summary/history is the GW-level source.
"""
from io import StringIO
import pandas as pd
import requests

from .cache import Cache

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
KEEP = ["element", "name", "position", "team", "GW", "total_points", "minutes",
        "xP", "expected_goals", "expected_assists", "bps", "was_home", "opponent_team"]


def load_season_gws(season: str, cache: Cache, session=None) -> pd.DataFrame:
    slug = f"archive-{season}"
    cached = cache.newest(slug)
    if cached is not None:
        return pd.DataFrame(cached[0])
    resp = (session or requests).get(ARCHIVE_URL.format(season=season), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df = df[[c for c in KEEP if c in df.columns]]
    cache.put(slug, df.to_dict("records"))
    cache.prune(slug, keep=3)
    return df


def verify_archive_integrity(gw_df: pd.DataFrame, past_df: pd.DataFrame,
                             season: str, tolerance: float = 0.02) -> dict:
    """Do per-GW rows sum to the API's season totals? If not, the archive is suspect."""
    totals = gw_df.groupby("element")["total_points"].sum()
    past = past_df[past_df["season_name"] == season].set_index("player_id")["total_points"]
    common = totals.index.intersection(past.index)
    checked = mismatched = 0
    for pid in common:
        checked += 1
        expected, actual = float(past.loc[pid]), float(totals.loc[pid])
        if abs(expected - actual) > max(1.0, tolerance * abs(expected)):
            mismatched += 1
    return {"checked": checked, "mismatched": mismatched,
            "ok": checked > 0 and mismatched == 0}
