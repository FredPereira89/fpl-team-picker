"""Validate the simplified BPS/bonus model against REAL historical fixtures.

Reads cached element-summary history (real minutes/goals/assists/cards/
saves/DC/conceded per player per fixture) and the real official `bps`/
`bonus` FPL awarded, computes this codebase's approximate BPS the same way
`fpl.model.bps.score_side_bps` does, ranks each fixture with
`award_match_bonus`, and reports how well the approximation's bonus
recipients and BPS values agree with the real ones -- exactly the kind of
predictive check unit tests on ranking MECHANICS cannot provide (Codex's
review-7 re-review, finding 3).

Usage: python scripts/validate_bps.py
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from fpl.model.bps import BPS_GOAL, BPS_ASSIST, BPS_SAVE, BPS_CARD, BPS_RED_CARD, \
    BPS_CONCEDED, BPS_APPEARANCE_SHORT, BPS_APPEARANCE_LONG, BPS_DC_ACTION, \
    CONCEDED_BPS_POSITIONS, award_match_bonus

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def _load_positions():
    files = sorted(glob.glob(str(CACHE / "bootstrap-static_*.json")))
    d = json.load(open(files[-1], encoding="utf-8"))
    pos_of_type = {et["id"]: et["singular_name_short"] for et in d["element_types"]}
    return {int(e["id"]): pos_of_type[int(e["element_type"])] for e in d["elements"]}


def _load_history():
    """One row per (player, fixture), from only the SINGLE newest cached
    element-summary snapshot per player -- each snapshot already contains
    the player's full history to date, so loading more than one file per
    player triples up every round they both cover."""
    newest_path = {}
    for path in glob.glob(str(CACHE / "element-summary-*.json")):
        pid = int(Path(path).stem.split("-")[-1].split("_")[0])
        ts = Path(path).stem.split("_")[-1]
        if pid not in newest_path or ts > newest_path[pid][0]:
            newest_path[pid] = (ts, path)

    rows = []
    for pid, (_, path) in newest_path.items():
        try:
            d = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for h in d.get("history", []):
            h = dict(h)
            h["player_id"] = pid
            rows.append(h)
    return pd.DataFrame(rows)


def _approx_bps(row, position, dc_weights=None, save_value=None) -> float:
    """`dc_weights` defaults to the SHIPPED production `BPS_DC_ACTION`;
    `logo_cv` passes candidate weight sets during cross-validation instead."""
    dc_weights = BPS_DC_ACTION if dc_weights is None else dc_weights
    save_value = BPS_SAVE if save_value is None else float(save_value)
    minutes = float(row["minutes"])
    played = minutes > 0
    reached_60 = minutes >= 60
    goals = float(row["goals_scored"])
    assists = float(row["assists"])
    saves = float(row["saves"])
    yellow_cards = float(row["yellow_cards"])
    red_cards = float(row["red_cards"])
    # Production dc90 (model.scoring.RATE_SPECS) is sourced from FPL's own
    # `defensive_contribution` field directly -- NOT a sum of the three raw
    # actions. That field is position-dependent and NOT their sum: it is 0
    # for GKP (not DC-threshold-eligible), CBI+tackles only for DEF
    # (recoveries excluded), and does include recoveries for MID/FWD.
    # Using the raw sum here validated a feature the shipped model never
    # actually sees for GKP/DEF (confirmed: GKP 815 vs 0, DEF 3771 vs 2584
    # summed across the cached GW1-4 history).
    dc = float(row.get("defensive_contribution", 0) or 0)
    conceded_on = float(row["goals_conceded"])
    # FPL's own `clean_sheets` field already implements the real rule
    # correctly (no goal conceded WHILE ON THE PITCH, 60+ minutes) --
    # confirmed via the official rules page: a player subbed at 60' keeps
    # his clean sheet even if his team concedes after he leaves. An
    # earlier version of this validator re-derived "clean sheet" from the
    # match's FINAL score instead, which is the same bug production had
    # (see model.simulate's own fix, same session) -- using the real field
    # here is both simpler and correct, rather than re-deriving it wrong.
    clean_sheet = float(row.get("clean_sheets", 0) or 0) > 0

    if not played:
        return -np.inf
    bps = BPS_APPEARANCE_LONG if reached_60 else BPS_APPEARANCE_SHORT
    bps += goals * BPS_GOAL.get(position, 0.0) + assists * BPS_ASSIST
    if position in ("GKP", "DEF") and reached_60 and clean_sheet:
        bps += 12.0
    bps += saves * save_value
    bps += yellow_cards * BPS_CARD + red_cards * BPS_RED_CARD
    bps += dc * dc_weights.get(position, 0.6)
    if position in CONCEDED_BPS_POSITIONS:
        bps += conceded_on * BPS_CONCEDED
    return bps


def _mae(hist, positions, dc_weights, save_value=None) -> float:
    errs = []
    for _, r in hist.iterrows():
        if float(r["minutes"]) <= 0:
            continue
        p = positions.get(int(r["player_id"]), "MID")
        errs.append(abs(_approx_bps(r, p, dc_weights, save_value) - float(r["bps"])))
    return float(np.mean(errs)) if errs else float("nan")


def fit_gkp_save_value(hist, positions, dc_weights=None, base_save=2.0) -> dict:
    """Fit the extra per-save BPS term on played goalkeeper rows.

    The response is ``real BPS - approximate BPS at base_save`` and the
    regression includes an intercept, so the slope estimates the part of the
    missing BPS that scales with saves without forcing the separate zero-save
    residual (passing accuracy and other unavailable events) into the save
    coefficient. The 95% interval uses HC3 heteroscedasticity-robust standard
    errors. This remains exploratory observational calibration, not a causal
    decomposition; fresh gameweeks are the prospective validation set.
    """
    dc_weights = BPS_DC_ACTION if dc_weights is None else dc_weights
    played = hist[hist["minutes"].astype(float) > 0]
    gkp = played[played["player_id"].map(positions).fillna("MID") == "GKP"]
    if len(gkp) < 3:
        raise ValueError("at least three played goalkeeper rows are required")

    saves = gkp["saves"].astype(float).to_numpy()
    if np.ptp(saves) <= 0:
        raise ValueError("goalkeeper save counts must vary to fit a slope")
    target = np.array([
        float(r["bps"]) - _approx_bps(r, "GKP", dc_weights, base_save)
        for _, r in gkp.iterrows()
    ])
    design = np.column_stack([np.ones(len(saves)), saves])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ target
    residual = target - design @ beta

    # HC3 sandwich covariance: robust to the clear increase in residual
    # spread across save counts without adding statsmodels as a dependency.
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    adjusted = residual / np.maximum(1.0 - leverage, 1e-12)
    meat = design.T @ (design * adjusted[:, None] ** 2)
    covariance = xtx_inv @ meat @ xtx_inv
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    dof = len(saves) - design.shape[1]
    critical = float(student_t.ppf(0.975, dof))
    slope = float(beta[1])
    corr = float(np.corrcoef(saves, target)[0, 1])
    return {
        "n": int(len(saves)),
        "base_save": float(base_save),
        "intercept": float(beta[0]),
        "extra_per_save": slope,
        "fitted_save": float(base_save) + slope,
        "slope_se_hc3": slope_se,
        "ci_low": slope - critical * slope_se,
        "ci_high": slope + critical * slope_se,
        "correlation": corr,
    }


# A handful of candidate per-position DC weight sets to grid-search over --
# small and hand-picked (not a fine continuous search). This grid was
# itself designed after looking at all 4 cached gameweeks, so `logo_cv`
# below is EXPLORATORY/grouped cross-validation, not a genuinely
# independent test: it shows whether the selection is stable across folds
# GIVEN this hypothesis space, not that the space itself was chosen
# without seeing the data. A real out-of-sample check needs gameweeks this
# grid was never informed by.
_DC_CANDIDATES = [
    {"GKP": 0.6, "DEF": 0.6, "MID": 0.6, "FWD": 0.6},
    {"GKP": 0.5, "DEF": 0.7, "MID": 0.8, "FWD": 0.5},
    {"GKP": 0.4, "DEF": 0.8, "MID": 0.9, "FWD": 0.6},
    {"GKP": 0.5, "DEF": 0.75, "MID": 0.85, "FWD": 0.55},
]


def logo_cv(hist, positions):
    """Leave-one-gameweek-out CV for the DC weights and save calibration.

    The save effect is re-fitted on each training partition before either
    the candidate DC weights or the held-out rows are scored. This prevents
    the held-out gameweek from leaking into a global BPS_SAVE estimate.

    (Codex's re-review, finding 2: the weights shipped in BPS_DC_ACTION
    were selected AND evaluated on the same 4 gameweeks -- in-sample model
    selection, not independent validation).

    For each held-out round, the BEST candidate on the OTHER rounds is
    picked and scored on the held-out one; this reports whether that
    selection is STABLE across folds and roughly matches its own in-
    sample number, not whether the whole exercise is independent of the
    data -- it is not, since `_DC_CANDIDATES` was designed after looking
    at all 4 gameweeks (exploratory/grouped CV, see that grid's comment).
    Genuinely fresh, untouched gameweeks are the real prospective test.
    """
    rounds = sorted(hist["round"].dropna().unique())
    print(f"\nleave-one-gameweek-out cross-validation ({len(rounds)} folds):")
    held_out_maes = []
    uniform_maes = []
    folds = []
    uniform = {"GKP": 0.6, "DEF": 0.6, "MID": 0.6, "FWD": 0.6}
    for held_out in rounds:
        train = hist[hist["round"] != held_out]
        test = hist[hist["round"] == held_out]
        save_fit = fit_gkp_save_value(train, positions)
        fold_save_value = save_fit["fitted_save"]
        best_weights, best_train_mae = None, float("inf")
        for cand in _DC_CANDIDATES:
            train_mae = _mae(train, positions, cand, fold_save_value)
            if train_mae < best_train_mae:
                best_train_mae, best_weights = train_mae, cand
        held_out_mae = _mae(test, positions, best_weights, fold_save_value)
        uniform_mae = _mae(test, positions, uniform, fold_save_value)
        held_out_maes.append(held_out_mae)
        uniform_maes.append(uniform_mae)
        folds.append({"held_out": int(held_out), "save_value": fold_save_value,
                      "weights": best_weights, "train_mae": best_train_mae,
                      "held_out_mae": held_out_mae, "uniform_mae": uniform_mae})
        print(f"  held out GW{int(held_out)}: selected {best_weights} "
              f"and save={fold_save_value:.2f} from training only "
              f"(train MAE {best_train_mae:.2f}) -> held-out MAE {held_out_mae:.2f}")
    print(f"  mean held-out MAE across folds: {np.mean(held_out_maes):.2f}")
    print(f"  for comparison, uniform 0.6 (no per-position fit) scored "
          f"{np.mean(uniform_maes):.2f} on the same folds")
    return {"folds": folds, "mean_held_out_mae": float(np.mean(held_out_maes)),
            "mean_uniform_mae": float(np.mean(uniform_maes))}


def main():
    positions = _load_positions()
    hist = _load_history()
    hist = hist[hist["fixture"].notna()]
    print(f"{len(hist)} player-fixture rows loaded, "
         f"{hist['player_id'].nunique()} players, {hist['fixture'].nunique()} fixtures")

    exact_match = 0
    top3_jaccard = []
    bps_abs_err = []
    err_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    bonus_recipient_recall = []   # of the REAL recipients, how many did approx also flag
    n_fixtures = 0
    for fixture_id, grp in hist.groupby("fixture"):
        grp = grp[grp["minutes"] > 0]
        if len(grp) < 2:
            continue
        n_fixtures += 1
        pos_list = [positions.get(int(pid), "MID") for pid in grp["player_id"]]
        approx = np.array([_approx_bps(r, p) for (_, r), p in zip(grp.iterrows(), pos_list)])
        real_bps = grp["bps"].to_numpy(dtype=float)
        real_bonus = grp["bonus"].to_numpy(dtype=float)
        errs = np.abs(approx - real_bps)
        bps_abs_err.extend(errs.tolist())
        for p, e in zip(pos_list, errs):
            err_by_pos[p].append(e)

        approx_bonus = award_match_bonus(approx[:, None])[:, 0]
        real_recipients = set(grp["player_id"].to_numpy()[real_bonus > 0])
        approx_recipients = set(grp["player_id"].to_numpy()[approx_bonus > 0])
        if real_recipients == approx_recipients:
            exact_match += 1
        union = real_recipients | approx_recipients
        inter = real_recipients & approx_recipients
        top3_jaccard.append(len(inter) / len(union) if union else 1.0)
        if real_recipients:
            bonus_recipient_recall.append(len(inter) / len(real_recipients))

    print(f"fixtures scored: {n_fixtures}")
    print(f"exact bonus-recipient-set match: {exact_match}/{n_fixtures} "
         f"({exact_match / n_fixtures:.1%})")
    print(f"mean Jaccard(bonus recipients): {np.mean(top3_jaccard):.3f}")
    print(f"mean recall (of real recipients, share also flagged by approx): "
         f"{np.mean(bonus_recipient_recall):.3f}")
    print(f"BPS mean absolute error: {np.mean(bps_abs_err):.2f}")
    print(f"BPS median absolute error: {np.median(bps_abs_err):.2f}")
    print()
    print("BPS mean absolute error by position:")
    for pos, errs in err_by_pos.items():
        if errs:
            print(f"  {pos}: n={len(errs)} MAE={np.mean(errs):.2f} median={np.median(errs):.2f}")

    save_fit = fit_gkp_save_value(hist, positions)
    print()
    print("exploratory goalkeeper save calibration (full available sample):")
    print(f"  n={save_fit['n']}, residual/save r={save_fit['correlation']:.3f}")
    print(f"  extra BPS per save={save_fit['extra_per_save']:.3f} "
          f"(HC3 95% CI {save_fit['ci_low']:.3f} to {save_fit['ci_high']:.3f})")
    print(f"  fitted total save value={save_fit['fitted_save']:.3f}; "
          f"shipped BPS_SAVE={BPS_SAVE:.3f}")

    logo_cv(hist, positions)


if __name__ == "__main__":
    main()
