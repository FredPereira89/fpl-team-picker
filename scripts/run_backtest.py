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
    # Pooled numbers can hide a model that only works on one cutoff, and the
    # replay now produces a target season at a time, so report each of them.
    for season, stats in sorted(result.get("by_cutoff", {}).items()):
        print(f"    cutoff {season}: MAE={stats['mae']:.2f}  n={stats['n']}")
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


# Matches fpl.model.minutes.TEAM_GAMES. Expected minutes per gameweek is last
# season's total spread over the season, which is leak-free by construction.
TEAM_GAMES_PER_SEASON = 38


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
    # Expected minutes from the TRAINING season only. The harness used to
    # multiply each prediction by the minutes that actually occurred, which
    # hands the rate model future playing time -- the single thing a forecast
    # most needs to get right, supplied for free.
    train_season["exp_minutes"] = (train_season["minutes"].astype(float)
                                   / float(TEAM_GAMES_PER_SEASON))
    rates = train_season.set_index("element")[["pts90_shrunk", "pts90_naive",
                                               "exp_minutes"]]

    # Every registered player-gameweek row, nonappearances included. Filtering
    # to minutes>0 deleted the largest source of FPL error and flattered the
    # model precisely where weekly selection is hardest.
    rows = test.merge(rates, left_on="element", right_index=True, how="inner")
    n_absent = int((rows["minutes"] <= 0).sum())
    print(f"  {len(rows)} 2025-26 GW rows with known 2024-25 history "
          f"({n_absent} of them nonappearances, previously discarded)")

    rows["pred_model"] = rows["pts90_shrunk"] * rows["exp_minutes"] / 90.0
    rows["pred_naive"] = rows["pts90_naive"] * rows["exp_minutes"] / 90.0
    rows["pred_fpl_xp"] = pd.to_numeric(rows["xP"], errors="coerce")

    positions = rows["position"].replace({"GK": "GKP"})
    model_metrics = evaluate_predictions(rows["pred_model"], rows["total_points"], positions)
    naive_metrics = evaluate_predictions(rows["pred_naive"], rows["total_points"], positions)
    fpl_metrics = evaluate_predictions(rows["pred_fpl_xp"].fillna(0), rows["total_points"], positions)

    print(f"\n  Model  : MAE={model_metrics['mae']:.2f}  Spearman(overall)={model_metrics['spearman_overall']:.3f}  n={model_metrics['n']}")
    print(f"  Naive  : MAE={naive_metrics['mae']:.2f}  Spearman(overall)={naive_metrics['spearman_overall']:.3f}")
    print(f"  FPL xP : MAE={fpl_metrics['mae']:.2f}  Spearman(overall)={fpl_metrics['spearman_overall']:.3f}")
    print(f"\n  Per-position Spearman (model): {model_metrics['spearman_by_position']}")
    print(f"  Per-position Spearman (naive): {naive_metrics['spearman_by_position']}")
    print(f"  Per-position Spearman (FPL xP): {fpl_metrics['spearman_by_position']}")

    # For continuity with every earlier run of this script, and clearly labelled
    # as what it is: the same comparison with future minutes supplied and
    # nonappearances removed. It is not evidence about the production model.
    appeared = rows[rows["minutes"] > 0].copy()
    appeared["pred_oracle_minutes"] = (appeared["pts90_shrunk"]
                                       * appeared["minutes"] / 90.0)
    oracle = evaluate_predictions(appeared["pred_oracle_minutes"],
                                  appeared["total_points"],
                                  appeared["position"].replace({"GK": "GKP"}))
    print("\n  [component diagnostic, rate model only — given the minutes that "
          "actually occurred, nonappearances removed]")
    print(f"  Model  : MAE={oracle['mae']:.2f}  "
          f"Spearman(overall)={oracle['spearman_overall']:.3f}  n={oracle['n']}")

    # full_pipeline=False on purpose: this harness scores a points-per-90 proxy
    # against archived rows. It never runs the production minutes model, the
    # component scoring split or the optimizer, so it cannot grant trust.
    gate = trust_gate(model_metrics, naive_metrics, fpl_metrics, full_pipeline=False)
    print(f"\n  TRUST GATE: trusted={gate['trusted']}")
    print(f"  {gate['summary']}")
    return {"model": model_metrics, "naive": naive_metrics, "fpl_xp": fpl_metrics,
            "component_diagnostic": oracle, "gate": gate}


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
    print(f"Tier 2 (per-GW, 2025-26 held out): component diagnostic only — "
          f"trust gate = {'TRUSTED' if t2['gate']['trusted'] else 'NOT TRUSTED'}")
    print("  Production trust comes from the sequential replay "
          "(scripts/run_walkforward.py), not from this script.")
    if not t2['gate']['trusted']:
        print(f"  Failures: {t2['gate']['failures']}")


if __name__ == "__main__":
    main()
