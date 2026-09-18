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
        # EXPOSURE-weighted: the prior a small sample is shrunk toward is the
        # positional rate per 90 across all minutes actually played, not the
        # mean of per-player rates. An unweighted mean gave a one-minute cameo
        # -- a 90-per-90 figure or a zero -- the same vote as a 3,000-minute
        # season, and the priors it produced were noise from the fringe.
        played = tmp[tmp["_mins"] > 0]
        weighted = (played["_raw"] * played["_mins"]).groupby(played["position"]).sum()
        exposure = played["_mins"].groupby(played["position"]).sum()
        pos_mean = (weighted / exposure.replace(0.0, np.nan)).fillna(0.0)
        fallback = (float((played["_raw"] * played["_mins"]).sum() / played["_mins"].sum())
                    if played["_mins"].sum() > 0 else 0.0)
        means = tmp["position"].map(pos_mean).fillna(fallback).astype(float)
        out[name] = (mins * raw + k * means) / (mins + k)
    return pd.DataFrame(out).reset_index(drop=True)


def current_per90(current: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date per-90 rates, unshrunk.

    Deliberately not shrunk toward a positional mean: how much this season
    counts is decided once, by form_weight, from how many gameweeks are on
    record. Shrinking here as well would discount the same small sample twice.
    """
    df = current.copy()
    df["_cards"] = df["yellow_cards"] + 3 * df["red_cards"]
    mins = df["minutes"].astype(float)
    out = {"player_id": df["player_id"].astype(int),
           "gws_played": df["gws_played"].astype(int),
           "minutes": mins}
    for name, source in dict(RATE_SPECS, cards90="_cards").items():
        out[name] = np.where(mins > 0, df[source].astype(float) / np.maximum(mins, 1) * 90.0, 0.0)
    return pd.DataFrame(out).reset_index(drop=True)


RATE_COLUMNS = list(RATE_SPECS) + ["cards90"]


def ew_per90(rounds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Season-to-date per-90 rates, weighted toward recent matches.

    `current_per90` treats a goal in August exactly like a goal last Saturday.
    `model.form_half_life_gw` has been in the config since the first commit to
    say how fast that should decay, and until now nothing read it: the weight
    on "form" was decided only by how MANY gameweeks were on record, never by
    which ones. Weights halve every `form_half_life_gw` rounds, applied to both
    the stat and the minutes so the ratio stays a rate.

    Deliberately unshrunk, exactly like `current_per90`: how much this season
    counts at all is decided once, by `form_weight`.
    """
    df = rounds.copy()
    df["_cards"] = df["yellow_cards"] + 3 * df["red_cards"]
    half_life = max(float(cfg.form_half_life_gw), 1e-9)
    latest = float(df["round"].max())
    w = 0.5 ** ((latest - df["round"].astype(float)) / half_life)
    weighted_minutes = (w * df["minutes"].astype(float)).groupby(df["player_id"]).sum()

    out = {"player_id": weighted_minutes.index.astype(int),
           "gws_played": df.groupby("player_id")["round"].nunique().astype(int),
           # Raw minutes, unweighted: the form BLEND is capped by how much
           # football the rates rest on, which recency weighting must not shrink.
           "minutes": df.groupby("player_id")["minutes"].sum().astype(float)}
    for name, source in dict(RATE_SPECS, cards90="_cards").items():
        totals = (w * df[source].astype(float)).groupby(df["player_id"]).sum()
        out[name] = np.where(weighted_minutes > 0,
                             totals / np.maximum(weighted_minutes, 1e-9) * 90.0, 0.0)
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


def blended_rates(players: pd.DataFrame, current: pd.DataFrame | None, cfg,
                  rounds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Last season's shrunk baseline, blended with season-to-date output.

    Before 2026-08-27 this blend existed (blend_form / form_weight) but nothing
    called it, so the model ran entirely on the previous season and could not
    see the current one at all -- which is also why a summer signing with no
    prior Premier League row was rated at the positional mean forever.

    Given `rounds` (per-match history) the current-season side is weighted
    toward recent matches at `model.form_half_life_gw`; without it, every
    gameweek so far counts the same.
    """
    base = per90_rates(players, cfg)
    if rounds is not None and len(rounds):
        cur = ew_per90(rounds, cfg).set_index("player_id")
    elif current is not None and len(current):
        cur = current_per90(current).set_index("player_id")
    else:
        return apply_set_piece_roles(base, players, cfg)
    out = base.copy()
    for i, pid in enumerate(out["player_id"].astype(int)):
        if pid not in cur.index:
            continue
        gws = int(cur.loc[pid, "gws_played"])
        played = (float(cur.loc[pid, "minutes"]) if "minutes" in cur.columns else None)
        for col in RATE_COLUMNS:
            out.loc[out.index[i], col] = blend_form(
                float(base.loc[base.index[i], col]), float(cur.loc[pid, col]), gws, cfg,
                minutes_played=played,
            )
    return apply_set_piece_roles(out, players, cfg)


def ew_mean(values: list[float], half_life: float) -> float:
    """Exponentially-weighted mean. `values` is oldest-first, newest last."""
    if not values:
        return 0.0
    n = len(values)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest has age 0
    weights = 0.5 ** (ages / float(half_life))
    return float(np.dot(weights, np.asarray(values, dtype=float)) / weights.sum())


# A full match's worth of minutes. Form evidence is measured in matches
# actually played, and a gameweek on record in which the player barely
# featured is not a gameweek of evidence about his per-90 rates.
FORM_MINUTES_PER_GW = 90.0


def form_weight(gws_played: int, cfg, minutes_played: float | None = None) -> float:
    """How much this season's rates count against last season's.

    Ramps to `form_max_weight` over FORM_WINDOW_GWS gameweeks -- but counted
    in EFFECTIVE gameweeks, the lesser of rounds on record and minutes played
    over ninety. Counting rounds alone handed a player with six cameos of
    fifteen minutes the same 60% weight on his current rates as an
    ever-present, and a tiny sample then set his projection.
    """
    if gws_played <= 0:
        return 0.0
    effective = float(gws_played)
    if minutes_played is not None:
        effective = min(effective, float(minutes_played) / FORM_MINUTES_PER_GW)
    if effective <= 0:
        return 0.0
    return min(1.0, effective / FORM_WINDOW_GWS) * float(cfg.form_max_weight)


def blend_form(baseline: float, form_value: float, gws_played: int, cfg,
               minutes_played: float | None = None) -> float:
    w = form_weight(gws_played, cfg, minutes_played)
    return (1 - w) * float(baseline) + w * float(form_value)
