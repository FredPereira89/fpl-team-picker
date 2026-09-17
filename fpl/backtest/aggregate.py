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
    """Predict every eligible season's points-per-90 from strictly earlier ones.

    Two leaks made the old version useless as evidence. It predicted only each
    player's FINAL season, discarding most of the out-of-sample data; and both
    the shrinkage prior and the "naive" baseline were computed over the WHOLE
    frame, target seasons included. Shrinking toward a mean that already
    contains the answer, and comparing against a baseline that IS the mean of
    the answers, can approve a worse rate model -- which then affects every
    weekly decision the tool makes.

    So the frame is replayed by season cutoff: for target season `t`, both the
    player's history and the population prior come only from seasons before `t`.
    """
    df = past.sort_values(["player_id", "season_name"]).copy()
    df["pts90"] = df.apply(_pts90, axis=1)
    k = float(cfg.shrinkage_minutes)

    seasons = sorted(df["season_name"].unique())
    preds, actuals, naive, by_cutoff = [], [], [], {}

    for target in seasons[1:]:
        prior = df[df["season_name"] < target]
        if prior.empty:
            continue
        # The prior population, weighted by exposure so a one-minute cameo does
        # not count as much as a full season.
        prior_minutes = float(prior["minutes"].sum())
        pop_mean = (float((prior["pts90"] * prior["minutes"]).sum()) / prior_minutes
                    if prior_minutes > 0 else 0.0)

        season_preds, season_actuals = [], []
        for pid, group in df[df["season_name"] == target].groupby("player_id"):
            history = prior[prior["player_id"] == pid]
            if history.empty:
                continue          # no pre-cutoff history: nothing to predict from
            mins = float(history["minutes"].sum())
            weighted = float((history["pts90"] * history["minutes"]).sum())
            season_preds.append((weighted + k * pop_mean) / (mins + k))
            season_actuals.append(float(group["pts90"].iloc[0]))
            # The honest baseline: what a forecaster who knew only the past
            # would have guessed. Not the mean of the answers.
            naive.append(pop_mean)

        if not season_preds:
            continue
        preds += season_preds
        actuals += season_actuals
        by_cutoff[str(target)] = {
            "mae": float(np.mean(np.abs(np.array(season_preds)
                                        - np.array(season_actuals)))),
            "n": len(season_preds),
        }

    if not preds:
        return {"mae": 0.0, "rmse": 0.0, "spearman": 0.0, "n": 0,
                "naive_mae": 0.0, "beats_naive": False, "by_cutoff": {}}

    preds_a, actual_a, naive_a = np.array(preds), np.array(actuals), np.array(naive)
    mae = float(np.mean(np.abs(preds_a - actual_a)))
    rmse = float(np.sqrt(np.mean((preds_a - actual_a) ** 2)))
    rho = float(spearmanr(preds_a, actual_a).statistic) if len(preds_a) > 2 else 0.0
    naive_mae = float(np.mean(np.abs(naive_a - actual_a)))
    return {"mae": mae, "rmse": rmse, "spearman": 0.0 if np.isnan(rho) else rho,
            "n": len(preds), "naive_mae": naive_mae, "beats_naive": mae < naive_mae,
            "by_cutoff": by_cutoff}
