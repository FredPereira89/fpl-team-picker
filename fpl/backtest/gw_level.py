"""Tier 2 backtest: per-gameweek accuracy and the model trust gate.

Validates what aggregates cannot: fixture adjustment, form decay, captaincy.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TOP_N = 20


def _rho(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return 0.0
    r = spearmanr(a.values, b.values).statistic
    return 0.0 if np.isnan(r) else float(r)


def evaluate_predictions(pred: pd.Series, actual: pd.Series,
                         positions: pd.Series) -> dict:
    df = pd.DataFrame({"pred": pred, "actual": actual, "pos": positions}).dropna()
    err = df["pred"] - df["actual"]
    n_top = min(TOP_N, len(df))
    top_pred = set(df["pred"].nlargest(n_top).index)
    top_actual = set(df["actual"].nlargest(n_top).index)
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman_overall": _rho(df["pred"], df["actual"]),
        "spearman_by_position": {
            pos: _rho(g["pred"], g["actual"]) for pos, g in df.groupby("pos")
        },
        "top20_overlap": len(top_pred & top_actual) / n_top if n_top else 0.0,
        "n": int(len(df)),
    }


def captaincy_hit_rate(pred_by_gw: dict[int, pd.Series],
                       actual_by_gw: dict[int, pd.Series]) -> float:
    hits = total = 0
    for gw, pred in pred_by_gw.items():
        actual = actual_by_gw.get(gw)
        if actual is None or pred.empty:
            continue
        total += 1
        if pred.idxmax() == actual.idxmax():
            hits += 1
    return hits / total if total else 0.0


def trust_gate(model: dict, naive: dict, fpl_xp: dict) -> dict:
    """Model is trusted only if per-position rank correlation beats BOTH baselines,
    for every position that model, naive, and fpl_xp all cover. A position
    missing from any of the three inputs is an automatic failure, not a
    silent pass -- incomplete data must never produce a false "trusted"."""
    model_pos = model["spearman_by_position"]
    naive_pos = naive["spearman_by_position"]
    fpl_pos = fpl_xp["spearman_by_position"]
    all_positions = set(model_pos) | set(naive_pos) | set(fpl_pos)

    failures = []
    for pos in sorted(all_positions):
        if pos not in model_pos:
            failures.append(f"{pos}: no model evaluation available for this position")
            continue
        rho = model_pos[pos]
        if pos not in naive_pos:
            failures.append(f"{pos}: naive baseline missing, cannot confirm model beats it")
        elif rho <= naive_pos[pos]:
            failures.append(f"{pos}: rank correlation {rho:.3f} does not beat naive baseline")
        if pos not in fpl_pos:
            failures.append(f"{pos}: FPL xP baseline missing, cannot confirm model beats it")
        elif rho <= fpl_pos[pos]:
            failures.append(f"{pos}: rank correlation {rho:.3f} does not beat FPL's own xP")

    trusted = not failures
    summary = ("Model beats both baselines in every position — recommendations can be "
               "trusted at face value." if trusted else
               "LOW CONFIDENCE — " + "; ".join(failures))
    return {"trusted": trusted, "failures": failures, "summary": summary}
