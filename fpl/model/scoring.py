"""Per-90 scoring rates with shrinkage, and the baseline/form blend.

At GW1 there is no current-season gameweek data (spec §2), so form_weight
returns 0.0 and the model runs entirely on last season's baseline.
"""
import numpy as np
import pandas as pd

FORM_WINDOW_GWS = 6
RATE_SPECS = {
    "xg90": "expected_goals",
    "xa90": "expected_assists",
    "bonus90": "bonus",
    "dc90": "defensive_contribution",
    "saves90": "saves",
}


def per90_rates(players: pd.DataFrame, cfg) -> pd.DataFrame:
    k = float(cfg.shrinkage_minutes)
    df = players.copy()
    df["_cards"] = df["yellow_cards"] + 3 * df["red_cards"]
    mins = df["minutes"].astype(float)

    out = {"player_id": df["player_id"].astype(int)}
    specs = dict(RATE_SPECS, cards90="_cards")
    for name, source in specs.items():
        raw = np.where(mins > 0, df[source].astype(float) / np.maximum(mins, 1) * 90.0, 0.0)
        tmp = df.assign(_raw=raw, _mins=mins)
        pos_mean = tmp[tmp["_mins"] > 0].groupby("position")["_raw"].mean()
        fallback = float(tmp.loc[tmp["_mins"] > 0, "_raw"].mean() or 0.0)
        means = tmp["position"].map(pos_mean).fillna(fallback).astype(float)
        out[name] = (mins * raw + k * means) / (mins + k)
    return pd.DataFrame(out).reset_index(drop=True)


def ew_mean(values: list[float], half_life: float) -> float:
    """Exponentially-weighted mean. `values` is oldest-first, newest last."""
    if not values:
        return 0.0
    n = len(values)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest has age 0
    weights = 0.5 ** (ages / float(half_life))
    return float(np.dot(weights, np.asarray(values, dtype=float)) / weights.sum())


def form_weight(gws_played: int, cfg) -> float:
    if gws_played <= 0:
        return 0.0
    return min(1.0, gws_played / FORM_WINDOW_GWS) * float(cfg.form_max_weight)


def blend_form(baseline: float, form_value: float, gws_played: int, cfg) -> float:
    w = form_weight(gws_played, cfg)
    return (1 - w) * float(baseline) + w * float(form_value)
