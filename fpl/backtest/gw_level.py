"""Tier 2 backtest: per-gameweek accuracy and the model trust gate.

Validates what aggregates cannot: fixture adjustment, form decay, captaincy.

What it CANNOT validate is equally important and was being claimed anyway. The
Tier 2 harness scores a rate model on rows filtered to `minutes > 0`, with the
prediction multiplied by the minutes that actually occurred. That hands the
model future playing time and deletes every nonappearance -- the single largest
source of FPL error -- so it says nothing about the pipeline that picks a team.
`trust_gate` therefore requires its caller to state, explicitly, whether the
numbers came from the full pipeline; a component diagnostic can never grant
production trust however good its correlations look.
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
                       actual_by_gw: dict[int, pd.Series],
                       owned_by_gw: dict[int, list[int]] | None = None) -> float:
    """How often the armband went on the best available player.

    With `owned_by_gw` this is the real question: of the players the manager
    OWNED that week, did the model's pick outscore the rest of them? Without
    it, the comparison is over the whole pool -- which is a diagnostic of the
    projection's top end, not of a captaincy decision, because no manager could
    have captained a player they did not own. That pool-level reading was being
    reported as a captaincy hit rate.
    """
    hits = total = 0
    for gw, pred in pred_by_gw.items():
        actual = actual_by_gw.get(gw)
        if actual is None or pred.empty:
            continue
        owned = owned_by_gw.get(gw) if owned_by_gw else None
        if owned is not None:
            keep = [p for p in owned if p in pred.index and p in actual.index]
            if not keep:
                continue
            pred_slice, actual_slice = pred.loc[keep], actual.loc[keep]
        else:
            pred_slice, actual_slice = pred, actual
        total += 1
        if pred_slice.idxmax() == actual_slice.idxmax():
            hits += 1
    return hits / total if total else 0.0


def trust_gate(model: dict, naive: dict, fpl_xp: dict,
               full_pipeline: bool = False) -> dict:
    """Model is trusted only if per-position rank correlation beats BOTH baselines,
    for every position that model, naive, and fpl_xp all cover. A position
    missing from any of the three inputs is an automatic failure, not a
    silent pass -- incomplete data must never produce a false "trusted".

    `full_pipeline` must be asserted by the caller and defaults to False. The
    harness that has always fed this gate scores a rate proxy on rows where the
    player actually played, using the minutes he actually played: it cannot
    validate minutes, captaincy, fixture-level xP or anything end to end, so
    its verdict is a component diagnostic whatever the correlations say.
    Defaulting to False means a caller that has not thought about this gets the
    safe answer rather than an accidental pass.
    """
    if not full_pipeline:
        return {
            "trusted": False,
            "failures": ["these metrics come from a component diagnostic, not the "
                         "full pipeline, so they cannot establish production trust"],
            "summary": ("COMPONENT DIAGNOSTIC — these numbers score a rate proxy on "
                        "players who appeared, using the minutes they actually "
                        "played. They cannot validate minutes, captaincy or the "
                        "end-to-end forecast, so they do not grant trust however "
                        "strong they look."),
        }
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
