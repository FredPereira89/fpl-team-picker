"""Derived team attack/defence ratings.

FPL's strength_attack_* and strength_defence_* fields are zeroed in the live
2026/27 data, so ratings are computed from last season's goals for/against
(carried over in bootstrap). Teams without most of a Premier League season
behind them -- promoted clubs -- fall back to a prior scaled from
strength_overall_*, or to an optional odds provider.
"""
import pandas as pd

HOME_ATT = 1.10
AWAY_ATT = 0.90
# A club counts as established only if its squad carries most of a Premier
# League season between them. The old test was `>= 1 minute`, which read "does
# any player have PL history" -- but a promoted club fields several players who
# logged minutes elsewhere, so it passed, and its goals for/against were then
# divided by the league mean as if they were a full season. Coventry came up
# with 2,415 minutes across five players (6 for, 42 against) and rated out at
# att 0.13 / dfn 0.09: the best defence in the league by a factor of ten, which
# made MCI v COV Man City's hardest attacking fixture of GW3. A real squad logs
# 29,000-50,000 minutes; the promoted clubs logged 0-4,642, so the two groups
# are separated by a wide margin and the threshold does not need to be precise.
MIN_TEAM_MINUTES = 10_000
# Premier League goals conceded per team per match. Used when the baseline
# season carries no usable history at all.
DEFAULT_LEAGUE_GC = 1.35


def league_goals_per_team_match(players: pd.DataFrame, min_minutes: float = 900.0,
                                default: float = DEFAULT_LEAGUE_GC) -> float:
    """League-average goals conceded per team-match, from a baseline season.

    `att` and `dfn` below are ratios to the league mean and so centre on 1.0.
    They are not goal counts, and `model.fixtures` needs a real rate to turn
    them into one -- without it, expected goals conceded comes out near 1.0
    and exp(-1.0) puts the average clean sheet at 0.44 against a true rate
    near 0.27, over-rewarding every goalkeeper and defender in the pool.

    Measured per 90 rather than as a team total over 38: FPL records goals
    conceded only while a player is on the pitch, and no player features in
    every match, so season totals divided by 38 undercount by ~10%.
    """
    df = players[players["minutes"].astype(float) >= float(min_minutes)]
    if len(df) == 0:
        return float(default)
    per90 = df["goals_conceded"].astype(float) / (df["minutes"].astype(float) / 90.0)
    rate = float(per90.mean())
    return rate if rate > 0 else float(default)


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

    established = df["mins"] >= MIN_TEAM_MINUTES
    mean_gf = df.loc[established, "gf"].mean() if established.any() else 1.0
    mean_ga = df.loc[established, "ga"].mean() if established.any() else 1.0

    att, dfn, conf = [], [], []
    for _, row in df.iterrows():
        if row["mins"] >= MIN_TEAM_MINUTES and mean_gf > 0 and mean_ga > 0:
            att.append(row["gf"] / mean_gf)
            dfn.append(row["ga"] / mean_ga)
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
