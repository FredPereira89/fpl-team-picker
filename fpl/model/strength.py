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


# Squad-minutes of CURRENT-season football at which this season's attack and
# defence rates carry half the weight of last season's. A team records roughly
# 990 minutes a match, so this is about nine games -- late enough that a new
# manager, a rebuilt defence or a promoted side has shown what it is, early
# enough that the ratings are not still describing last May in October.
CURRENT_SEASON_SHRINK_MINUTES = 9000.0


def _rate_ratios(gf: pd.Series, ga: pd.Series, mins: pd.Series,
                 established: float) -> tuple[pd.Series, pd.Series]:
    """Attack and defence as ratios to the league mean, or NaN where unusable.

    Rates, not totals: summed season totals scale with how much football a
    squad played, so a club whose players have barely appeared reads as the
    meanest defence in the league.
    """
    per90 = (mins.astype(float) / 90.0).replace(0.0, float("nan"))
    gf90, ga90 = gf / per90, ga / per90
    solid = mins >= established
    if not solid.any():
        solid = mins >= MIN_MINUTES
    mean_gf90 = gf90[solid].mean() if solid.any() else float("nan")
    mean_ga90 = ga90[solid].mean() if solid.any() else float("nan")
    if not (mean_gf90 and mean_ga90) or mean_gf90 <= 0 or mean_ga90 <= 0:
        nan = pd.Series(float("nan"), index=gf.index)
        return nan, nan
    return gf90 / mean_gf90, ga90 / mean_ga90


def current_season_by_team(current: pd.DataFrame | None,
                           players: pd.DataFrame) -> pd.DataFrame | None:
    """Season-to-date goals for/against and minutes, per team.

    `history_current_frame` is keyed by player and carries no team, so the club
    comes from `players`. A player who has moved clubs mid-season is credited
    to the club he is at now, which is the club being rated.
    """
    if current is None or len(current) == 0:
        return None
    team_by_player = dict(zip(players["player_id"].astype(int),
                              players["team_id"].astype(int)))
    df = current.copy()
    df["team_id"] = df["player_id"].astype(int).map(team_by_player)
    df = df[df["team_id"].notna()]
    if len(df) == 0:
        return None
    return df.groupby(df["team_id"].astype(int)).agg(
        gf=("goals_scored", "sum"),
        ga=("goals_conceded", "sum"),
        mins=("minutes", "sum"),
    )


def team_ratings(players: pd.DataFrame, teams: pd.DataFrame, odds_provider=None,
                 current: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attack and defence multipliers, centred on 1.0.

    The baseline is the last completed season, shrunk toward FPL's own overall
    strength for clubs with little Premier League history. `current` (this
    season's per-player totals) is blended in as it accumulates: without it the
    ratings describe last season all the way to May, and underreact to a new
    manager, a rebuilt defence, a promoted side or a genuinely changed team.
    """
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
    att_ratio, dfn_ratio = _rate_ratios(df["gf"], df["ga"], df["mins"],
                                        ESTABLISHED_SQUAD_MINUTES)
    usable = bool(att_ratio.notna().any())

    now = current_season_by_team(current, players)
    if now is not None:
        now = now.reindex(df["team_id"].values).fillna({"gf": 0, "ga": 0, "mins": 0})
        now.index = df.index
        # This season is measured against THIS season's league average, so the
        # two ratio scales are comparable even a few games in, when goals per
        # game across the league is still noisy.
        cur_att, cur_dfn = _rate_ratios(now["gf"], now["ga"], now["mins"],
                                        CURRENT_SEASON_SHRINK_MINUTES / 3)

    att, dfn, conf = [], [], []
    for i, row in df.iterrows():
        overall = (row["strength_overall_home"] + row["strength_overall_away"]) / 2
        prior_att = _prior_from_overall(overall)
        prior_dfn = 2.0 - prior_att  # weak attack prior implies weak defence
        if row["mins"] >= MIN_MINUTES and usable and pd.notna(att_ratio[i]):
            # Shrink toward the prior by how much football actually backs the
            # rate, the same way per-90 player rates and p_start are shrunk.
            w = float(row["mins"]) / (float(row["mins"]) + SHRINK_MINUTES)
            a = w * att_ratio[i] + (1 - w) * prior_att
            d = w * dfn_ratio[i] + (1 - w) * prior_dfn
            confidence = "high" if w >= 0.5 else "low"
        else:
            a, d, confidence = prior_att, prior_dfn, "low"

        if now is not None and pd.notna(cur_att.get(i, float("nan"))):
            cur_mins = float(now.loc[i, "mins"])
            w_now = cur_mins / (cur_mins + CURRENT_SEASON_SHRINK_MINUTES)
            a = (1 - w_now) * a + w_now * float(cur_att[i])
            d = (1 - w_now) * d + w_now * float(cur_dfn[i])
            if w_now >= 0.5:
                confidence = "high"

        att.append(a)
        dfn.append(d)
        conf.append(confidence)
    df["att"], df["dfn"], df["confidence"] = att, dfn, conf

    if odds_provider is not None:
        factors = odds_provider.team_factors(list(df["team_id"]))
        for team_id, (a, d) in factors.items():
            mask = df["team_id"] == team_id
            df.loc[mask, ["att", "dfn", "confidence"]] = [a, d, "high"]

    return df[["team_id", "att", "dfn", "confidence"]].reset_index(drop=True)
