#!/usr/bin/env python3
"""Run Tier 1 (multi-season aggregate) and Tier 2 (per-GW) backtests against
real data, and report the model's trust-gate verdict.

Tier 1 tests the shrunk per-90 baseline across up to 8 real prior seasons per
player (fpl.backtest.aggregate.walk_forward_aggregate).

Tier 2 tests specifically the SAME shrinkage baseline's per-gameweek accuracy,
held out on a real season it never saw (predict 2025-26 from 2024-25 alone).
It does NOT test fixture adjustment or the minutes-probability model in
isolation -- those would need the full per-GW pipeline re-run retroactively,
which is a larger undertaking. This is an honest, narrower slice: exactly the
component that is 100% of the model at GW1, since form_weight=0 until GW6.

Predictions are scaled by each player's ACTUAL minutes that gameweek (not a
predicted minutes probability), so this isolates the per-90 rate quality
from minutes forecasting -- the two are validated separately by design.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from fpl.config import load_config
from fpl.data.cache import Cache
from fpl.data.client import FplClient
from fpl.data.archive import load_season_gws
from fpl.backtest.aggregate import walk_forward_aggregate
from fpl.backtest.gw_level import evaluate_predictions, trust_gate

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
SAMPLE_SIZE = 250  # players sampled for Tier 1 (rate-limited element-summary calls)


def tier1(client: FplClient, cache: Cache) -> dict:
    print(f"\n=== Tier 1: multi-season aggregate backtest (sample of {SAMPLE_SIZE} players) ===")
    bootstrap = client.bootstrap()
    elements = sorted(bootstrap["elements"], key=lambda e: e.get("total_points", 0), reverse=True)
    sample = elements[:SAMPLE_SIZE]

    summaries = {}
    for i, el in enumerate(sample):
        pid = el["id"]
        cached = cache.newest(f"element-summary-{pid}")
        if cached is not None:
            summaries[pid] = cached[0]
        else:
            import requests
            resp = requests.get(
                f"https://fantasy.premierleague.com/api/element-summary/{pid}/",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
            )
            resp.raise_for_status()
            summaries[pid] = resp.json()
            cache.put(f"element-summary-{pid}", summaries[pid])
            time.sleep(1.0)
        if (i + 1) % 25 == 0:
            print(f"  fetched {i + 1}/{len(sample)} player histories...")

    from fpl.data.normalize import history_past_frame
    past = history_past_frame(summaries)
    print(f"  {len(past)} player-seasons across {past.player_id.nunique()} players")

    cfg = load_config(ROOT / "config.yaml")
    result = walk_forward_aggregate(past, cfg)
    print(f"  n={result['n']}  MAE={result['mae']:.2f}  RMSE={result['rmse']:.2f}  "
          f"Spearman={result['spearman']:.3f}")
    print(f"  naive MAE={result['naive_mae']:.2f}  beats_naive={result['beats_naive']}")
    return result


def _shrunk_pts90(df: pd.DataFrame, cfg) -> pd.Series:
    """Same shrinkage formula as fpl.model.scoring.per90_rates, applied to
    total_points instead of a single component rate."""
    mins = df["minutes"].astype(float)
    raw = np.where(mins > 0, df["total_points"].astype(float) / np.maximum(mins, 1) * 90.0, 0.0)
    tmp = df.assign(_raw=raw, _mins=mins)
    pos_mean = tmp[tmp["_mins"] > 0].groupby("position")["_raw"].mean()
    fallback = float(tmp.loc[tmp["_mins"] > 0, "_raw"].mean() or 0.0)
    means = tmp["position"].map(pos_mean).fillna(fallback).astype(float)
    k = float(cfg.shrinkage_minutes)
    return (mins * raw + k * means) / (mins + k)


def tier2(cache: Cache, cfg) -> dict:
    print("\n=== Tier 2: per-GW backtest (train on 2024-25, test on 2025-26) ===")
    train = load_season_gws("2024-25", cache)
    test = load_season_gws("2025-26", cache)
    print(f"  2024-25: {len(train)} rows, {train.element.nunique()} players")
    print(f"  2025-26: {len(test)} rows, {test.element.nunique()} players")

    train_season = (
        train.groupby(["element", "position"], as_index=False)
        .agg(total_points=("total_points", "sum"), minutes=("minutes", "sum"))
    )
    train_season["pts90_shrunk"] = _shrunk_pts90(train_season, cfg)
    train_season["pts90_naive"] = np.where(
        train_season["minutes"] > 0,
        train_season["total_points"] / train_season["minutes"].clip(lower=1) * 90.0,
        0.0,
    )
    rates = train_season.set_index("element")[["pts90_shrunk", "pts90_naive"]]

    played = test[test["minutes"] > 0].merge(
        rates, left_on="element", right_index=True, how="inner"
    )
    print(f"  {len(played)} 2025-26 GW rows with minutes>0 AND known 2024-25 history")

    played["pred_model"] = played["pts90_shrunk"] * played["minutes"] / 90.0
    played["pred_naive"] = played["pts90_naive"] * played["minutes"] / 90.0
    played["pred_fpl_xp"] = pd.to_numeric(played["xP"], errors="coerce")

    positions = played["position"].replace({"GK": "GKP"})
    model_metrics = evaluate_predictions(played["pred_model"], played["total_points"], positions)
    naive_metrics = evaluate_predictions(played["pred_naive"], played["total_points"], positions)
    fpl_metrics = evaluate_predictions(played["pred_fpl_xp"].fillna(0), played["total_points"], positions)

    print(f"\n  Model  : MAE={model_metrics['mae']:.2f}  Spearman(overall)={model_metrics['spearman_overall']:.3f}  n={model_metrics['n']}")
    print(f"  Naive  : MAE={naive_metrics['mae']:.2f}  Spearman(overall)={naive_metrics['spearman_overall']:.3f}")
    print(f"  FPL xP : MAE={fpl_metrics['mae']:.2f}  Spearman(overall)={fpl_metrics['spearman_overall']:.3f}")
    print(f"\n  Per-position Spearman (model): {model_metrics['spearman_by_position']}")
    print(f"  Per-position Spearman (naive): {naive_metrics['spearman_by_position']}")
    print(f"  Per-position Spearman (FPL xP): {fpl_metrics['spearman_by_position']}")

    gate = trust_gate(model_metrics, naive_metrics, fpl_metrics)
    print(f"\n  TRUST GATE: trusted={gate['trusted']}")
    print(f"  {gate['summary']}")
    return {"model": model_metrics, "naive": naive_metrics, "fpl_xp": fpl_metrics, "gate": gate}


def main():
    cfg = load_config(ROOT / "config.yaml")
    cache = Cache(DATA_ROOT / "cache")
    client = FplClient(cache, ttl_hours=cfg.cache_ttl_hours)

    t1 = tier1(client, cache)
    t2 = tier2(cache, cfg)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tier 1 (multi-season aggregate): beats naive = {t1['beats_naive']} "
          f"(MAE {t1['mae']:.2f} vs naive {t1['naive_mae']:.2f}, n={t1['n']})")
    print(f"Tier 2 (per-GW, 2025-26 held out): trust gate = "
          f"{'TRUSTED' if t2['gate']['trusted'] else 'NOT TRUSTED'}")
    if not t2['gate']['trusted']:
        print(f"  Failures: {t2['gate']['failures']}")


if __name__ == "__main__":
    main()
