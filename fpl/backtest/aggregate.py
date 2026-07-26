"""Tier 1 backtest: multi-season walk-forward over history_past aggregates.

Validates the per-90 baseline and shrinkage. Cannot validate fixture
adjustment, form decay, or captaincy — that is Tier 2's job.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _pts90(row) -> float:
    return float(row["total_points"]) / max(float(row["minutes"]), 1.0) * 90.0


def walk_forward_aggregate(past: pd.DataFrame, cfg) -> dict:
    """Predict each season's points-per-90 from the player's prior seasons."""
    df = past.sort_values(["player_id", "season_name"]).copy()
    df["pts90"] = df.apply(_pts90, axis=1)

    preds, actuals = [], []
    for _, group in df.groupby("player_id"):
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        history = rows[:-1]
        target = rows[-1]
        mins = sum(float(r["minutes"]) for r in history)
        weighted = sum(_pts90(r) * float(r["minutes"]) for r in history)
        k = float(cfg.shrinkage_minutes)
        pop_mean = float(df["pts90"].mean())
        pred = (weighted + k * pop_mean) / (mins + k)
        preds.append(pred)
        actuals.append(float(target["pts90"]))

    if not preds:
        return {"mae": 0.0, "rmse": 0.0, "spearman": 0.0, "n": 0,
                "naive_mae": 0.0, "beats_naive": False}

    preds_a, actual_a = np.array(preds), np.array(actuals)
    mae = float(np.mean(np.abs(preds_a - actual_a)))
    rmse = float(np.sqrt(np.mean((preds_a - actual_a) ** 2)))
    rho = float(spearmanr(preds_a, actual_a).statistic) if len(preds_a) > 2 else 0.0
    naive = float(np.mean(np.abs(np.full_like(actual_a, actual_a.mean()) - actual_a)))
    return {"mae": mae, "rmse": rmse, "spearman": 0.0 if np.isnan(rho) else rho,
            "n": len(preds), "naive_mae": naive, "beats_naive": mae < naive}
