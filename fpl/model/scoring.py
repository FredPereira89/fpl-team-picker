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


# What a designated taker is worth per 90, over and above open play. A club wins
# roughly one penalty every eight matches and converts about four in five.
PENALTY_XG90 = 0.10
CORNER_XA90 = 0.05
FREEKICK_XG90 = 0.02
SECOND_CHOICE_SHARE = 0.25  # the backup only takes them when the first is off
SET_PIECE_COLS = ["penalties_order", "corners_and_indirect_freekicks_order",
                  "direct_freekicks_order"]


def _duty_share(order) -> float:
    if pd.isna(order):
        return 0.0
    order = int(order)
    if order <= 1:
        return 1.0
    if order == 2:
        return SECOND_CHOICE_SHARE
    return 0.0


def apply_set_piece_roles(rates: pd.DataFrame, players: pd.DataFrame, cfg) -> pd.DataFrame:
    """Credit designated penalty, corner and free-kick takers.

    FPL publishes these roles and the model read none of them, even though set
    pieces are the largest single driver of a defender's or deep midfielder's
    assist rate and a penalty taker's goal rate.

    The premium is scaled by how much of a player's rate is still PRIOR rather
    than his own measured output: a player with a full season on record already
    has last year's penalties inside his xG90, and crediting him again would
    count them twice. A summer signing, whose rate is entirely a positional
    guess, gets the whole premium -- which is exactly where the signal is worth
    the most, because nothing else in the model knows he takes them.
    """
    out = rates.copy()
    if not any(c in players.columns for c in SET_PIECE_COLS):
        return out

    k = float(cfg.shrinkage_minutes)
    mins = players.set_index("player_id")["minutes"].astype(float)
    roles = players.set_index("player_id")

    for idx, pid in zip(out.index, out["player_id"].astype(int)):
        if pid not in roles.index:
            continue
        unmeasured = k / (float(mins.get(pid, 0.0)) + k)
        pens = _duty_share(roles.loc[pid].get("penalties_order"))
        corners = _duty_share(roles.loc[pid].get("corners_and_indirect_freekicks_order"))
        frees = _duty_share(roles.loc[pid].get("direct_freekicks_order"))
        out.loc[idx, "xg90"] += unmeasured * (pens * PENALTY_XG90 + frees * FREEKICK_XG90)
        out.loc[idx, "xa90"] += unmeasured * corners * CORNER_XA90
    return out


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
