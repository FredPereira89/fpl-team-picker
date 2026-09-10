"""Recalibrate xP against what actually happened, per position.

The 2026-09-09 statistical review found the projection well calibrated
POOL-WIDE -- bias -0.08 pts per player-gameweek, 95% CI [-0.18, +0.02] over
1,890 player-gameweeks -- while differing BY POSITION (GKP +0.72, MID -0.16 on
GW2). Position is the axis that matters, because the optimizer is choosing
between positions under a fixed budget and fixed squad shape.

Why per position and not a single global fit: an affine transform applied
uniformly cannot change any decision. Every squad has the same fifteen slots,
so `a + b*xp` with `b > 0` shifts and scales every candidate squad's objective
identically and leaves the argmax exactly where it was. A global recalibration
would change the numbers printed in the report and nothing else. Only a
correction that differs ACROSS players -- here, across positions -- moves picks.

The fit is deliberately conservative:

* it refuses below `MIN_GAMEWEEKS`, because a slope fitted on two gameweeks is
  noise dressed as a correction;
* each position is shrunk toward the pooled fit by how much evidence backs it,
  so a thin position cannot acquire a wild slope of its own;
* it must be fitted WALK-FORWARD (on gameweeks strictly before the one being
  predicted), or it is scoring the model on its own answers.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .xp import EVENT_PREFIX

# Below this the correction is not distinguishable from noise. The review's
# sensitivity analysis put the smallest detectable weekly edge at ~13 pts on
# ten gameweeks, so this is already a generous floor.
MIN_GAMEWEEKS = 5
# Player-gameweek observations at which a position's own fit outweighs the
# pooled one. Roughly a position's worth of a short season.
SHRINK_OBSERVATIONS = 400.0
PROJECTION_PREFIXES = ("xp_next", "xp_horizon", "xp_gw")


@dataclass
class Calibration:
    intercept: dict
    slope: dict
    pooled_intercept: float
    pooled_slope: float
    n_gameweeks: int
    n_observations: int
    n_by_position: dict
    r2: float
    summary: str = ""


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(intercept, slope) of y on x; falls back to the identity if degenerate."""
    if len(x) < 3 or np.std(x) < 1e-9:
        return 0.0, 1.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def fit_calibration(scored: pd.DataFrame, min_gameweeks: int = MIN_GAMEWEEKS):
    """Fit `actual ~ intercept + slope * xp` per position, or None if too thin.

    `scored` needs `gw`, `position`, `xp_next1` and `actual`: one row per
    player-gameweek that has actually been played.
    """
    if scored is None or len(scored) == 0:
        return None
    n_gw = int(scored["gw"].nunique())
    if n_gw < int(min_gameweeks):
        return None

    x_all = scored["xp_next1"].astype(float).to_numpy()
    y_all = scored["actual"].astype(float).to_numpy()
    p_int, p_slope = _ols(x_all, y_all)

    intercept, slope, n_by = {}, {}, {}
    for pos, g in scored.groupby("position"):
        x = g["xp_next1"].astype(float).to_numpy()
        y = g["actual"].astype(float).to_numpy()
        i_pos, s_pos = _ols(x, y)
        # Shrink toward the pooled fit by how much football backs this position,
        # the same way every other rate in this model is shrunk.
        w = len(g) / (len(g) + SHRINK_OBSERVATIONS)
        intercept[str(pos)] = w * i_pos + (1 - w) * p_int
        slope[str(pos)] = w * s_pos + (1 - w) * p_slope
        n_by[str(pos)] = int(len(g))

    fitted = p_int + p_slope * x_all
    ss_res = float(np.sum((y_all - fitted) ** 2))
    ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)

    detail = ", ".join(f"{p} {slope[p]:.2f}x{intercept[p]:+.2f}" for p in sorted(slope))
    summary = (f"Calibrated on {n_gw} gameweeks ({len(scored)} player-gameweeks, "
               f"R2 {r2:.3f}). Pooled {p_slope:.2f}x{p_int:+.2f}; by position: {detail}.")
    return Calibration(intercept=intercept, slope=slope, pooled_intercept=p_int,
                       pooled_slope=p_slope, n_gameweeks=n_gw,
                       n_observations=int(len(scored)), n_by_position=n_by,
                       r2=r2, summary=summary)


def projection_columns(xp: pd.DataFrame) -> list[str]:
    return [c for c in xp.columns
            if any(str(c).startswith(p) for p in PROJECTION_PREFIXES)]


def apply_calibration(xp: pd.DataFrame, cal: Calibration | None) -> pd.DataFrame:
    """Rescale every projection column by its position's fitted transform.

    All projection columns move together: `xp_next1`, `xp_next5`, `xp_horizon`
    and each per-gameweek column are the same quantity over different windows,
    so calibrating one and not the others would let the optimizer and the
    report disagree about the same player.

    The intercept is scaled with the window -- a per-match offset applies once
    per match -- and the result is floored at zero, since no projection of an
    FPL return is meaningfully negative.
    """
    if cal is None:
        return xp
    out = xp.copy()
    pos = out["position"].astype(str)
    slope = pos.map(cal.slope).fillna(cal.pooled_slope).astype(float)
    inter = pos.map(cal.intercept).fillna(cal.pooled_intercept).astype(float)
    base = out["xp_next1"].astype(float).replace(0.0, np.nan)
    # The true ceiling on how many matches any column can cover: the number of
    # per-event columns actually in this frame. A rotation-risk player with a
    # fixture doubt THIS week (xp_next1 near zero) but a normal horizon
    # otherwise inflates the values/base ratio far past his real match count,
    # so bounding it to an arbitrary constant either over-applies the
    # per-match offset for him or truncates it for a genuinely long horizon.
    # Bounding to the real number of event columns fixes both.
    n_events = sum(1 for c in out.columns if str(c).startswith(EVENT_PREFIX))
    ceiling = float(n_events) if n_events else 20.0
    for col in projection_columns(out):
        values = out[col].astype(float)
        # How many matches this column covers, inferred from its size relative
        # to the single-gameweek projection, so the offset is not applied once
        # to a five-gameweek total.
        windows = (values / base).fillna(1.0).clip(lower=0.0, upper=ceiling)
        out[col] = (slope * values + inter * windows).clip(lower=0.0)
    return out


def scored_history(root, summaries: dict, before_event: int) -> pd.DataFrame:
    """Every past forecast joined to what actually happened, before `before_event`.

    This is the training set for the calibration, and it is assembled strictly
    walk-forward: a forecast is only usable once its gameweek has been played,
    and never for the gameweek about to be predicted. Fitting on the gameweek
    you are forecasting is scoring the model on its own answers.
    """
    from ..backtest.ledger import available_gameweeks, load_predictions
    from ..backtest.walkforward import actuals_frame

    actuals = actuals_frame(summaries)
    if len(actuals) == 0:
        return pd.DataFrame(columns=["gw", "player_id", "position", "xp_next1", "actual"])
    played = set(actuals["round"].astype(int))

    frames = []
    for gw in available_gameweeks(root):
        if gw >= int(before_event) or gw not in played:
            continue
        pred = load_predictions(gw, root)
        a = actuals[actuals["round"] == gw][["player_id", "actual", "minutes"]]
        merged = pred.merge(a, on="player_id", how="inner")
        if len(merged):
            merged["gw"] = int(gw)
            frames.append(merged[["gw", "player_id", "position", "xp_next1",
                                  "actual", "minutes"]])
    if not frames:
        return pd.DataFrame(columns=["gw", "player_id", "position", "xp_next1", "actual"])
    return pd.concat(frames, ignore_index=True)
