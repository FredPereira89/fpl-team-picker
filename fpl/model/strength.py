"""Derived team attack/defence ratings.

FPL's strength_attack_* and strength_defence_* fields are zeroed in the live
2026/27 data, so ratings are computed from last season's goals for/against
(carried over in bootstrap). Teams with no PL history fall back to a prior
scaled from strength_overall_*, or to an optional odds provider.
"""
import pandas as pd

HOME_ATT = 1.10
AWAY_ATT = 0.90
MIN_MINUTES = 1  # a team with zero recorded minutes has no PL history


def _prior_from_overall(strength: float) -> float:
    """Map FPL's 1-5 overall strength onto a multiplicative factor near 1.0."""
    return 0.70 + 0.15 * float(strength)


def team_ratings(players: pd.DataFrame, teams: pd.DataFrame, odds_provider=None) -> pd.DataFrame:
    agg = players.groupby("team_id").agg(
        gf=("goals_scored", "sum"),
        ga=("goals_conceded", "sum"),
        mins=("minutes", "sum"),
    )
    df = teams[["team_id", "strength_overall_home", "strength_overall_away"]].merge(
        agg, left_on="team_id", right_index=True, how="left"
    ).fillna({"gf": 0, "ga": 0, "mins": 0})

    established = df["mins"] >= MIN_MINUTES
    mean_gf = df["gf"].mean() if len(df) > 0 else 1.0
    mean_ga = df["ga"].mean() if len(df) > 0 else 1.0

    att, dfn, conf = [], [], []
    for _, row in df.iterrows():
        if row["mins"] >= MIN_MINUTES and mean_gf > 0 and mean_ga > 0:
            a = row["gf"] / mean_gf
            att.append(a)
            dfn.append(2.0 - a)  # symmetric: attack+defence~2.0 (centred on 1.0 each)
            conf.append("high")
        else:
            overall = (row["strength_overall_home"] + row["strength_overall_away"]) / 2
            a = _prior_from_overall(overall)
            att.append(a)
            dfn.append(2.0 - a)  # weak attack prior implies weak defence
            conf.append("low")
    df["att"], df["dfn"], df["confidence"] = att, dfn, conf

    if odds_provider is not None:
        factors = odds_provider.team_factors(list(df["team_id"]))
        for team_id, (a, d) in factors.items():
            mask = df["team_id"] == team_id
            df.loc[mask, ["att", "dfn", "confidence"]] = [a, d, "high"]

    return df[["team_id", "att", "dfn", "confidence"]].reset_index(drop=True)
