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
# Squad-minutes at which a team's own measured rate outweighs its strength_overall
# prior. Roughly a third of a full squad-season (a 20-man squad playing 38 games
# records ~65k minutes), so a promoted club with a few thousand sits near the
# prior while an established one is rated almost entirely on what it did.
SHRINK_MINUTES = 9000.0
# Squad-minutes a team needs before its rate helps define the league average.
# Promoted clubs' rates are built from too little football to centre anything.
ESTABLISHED_SQUAD_MINUTES = 9000.0
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

    # Rates, not totals. Summed season totals scale with how much football a
    # squad played, so a promoted club whose players have barely appeared in the
    # Premier League read as the meanest defence in it: Hull City rated 0.040
    # on 1,477 recorded minutes while Arsenal, on 41,084, rated 0.804. That put
    # their goalkeeper's clean-sheet probability at 0.91.
    per90 = (df["mins"].astype(float) / 90.0).replace(0.0, float("nan"))
    gf90, ga90 = df["gf"] / per90, df["ga"] / per90

    solid = df["mins"] >= ESTABLISHED_SQUAD_MINUTES
    if not solid.any():
        solid = df["mins"] >= MIN_MINUTES
    mean_gf90 = gf90[solid].mean() if solid.any() else float("nan")
    mean_ga90 = ga90[solid].mean() if solid.any() else float("nan")
    usable = bool(mean_gf90 and mean_ga90) and mean_gf90 > 0 and mean_ga90 > 0

    att, dfn, conf = [], [], []
    for i, row in df.iterrows():
        overall = (row["strength_overall_home"] + row["strength_overall_away"]) / 2
        prior_att = _prior_from_overall(overall)
        prior_dfn = 2.0 - prior_att  # weak attack prior implies weak defence
        if row["mins"] >= MIN_MINUTES and usable:
            # Shrink toward the prior by how much football actually backs the
            # rate, the same way per-90 player rates and p_start are shrunk.
            w = float(row["mins"]) / (float(row["mins"]) + SHRINK_MINUTES)
            att.append(w * (gf90[i] / mean_gf90) + (1 - w) * prior_att)
            dfn.append(w * (ga90[i] / mean_ga90) + (1 - w) * prior_dfn)
            conf.append("high" if w >= 0.5 else "low")
        else:
            att.append(prior_att)
            dfn.append(prior_dfn)
            conf.append("low")
    df["att"], df["dfn"], df["confidence"] = att, dfn, conf

    if odds_provider is not None:
        factors = odds_provider.team_factors(list(df["team_id"]))
        for team_id, (a, d) in factors.items():
            mask = df["team_id"] == team_id
            df.loc[mask, ["att", "dfn", "confidence"]] = [a, d, "high"]

    return df[["team_id", "att", "dfn", "confidence"]].reset_index(drop=True)
