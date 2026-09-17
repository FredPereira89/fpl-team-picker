import pytest
import pandas as pd
from fpl.config import Config
from fpl.model.minutes import minutes_model, M_START, M_SUB

CFG = Config(shrinkage_minutes=900, news_weight=0.5)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "web_name": ["Nailed", "Rotation", "Injured", "Doubt", "NewSigning"],
    "position": ["MID", "MID", "DEF", "FWD", "MID"],
    "team_id": [1, 1, 2, 2, 3],
    "price": [9.0, 5.0, 4.5, 7.0, 8.5],
    "status": ["a", "a", "i", "d", "a"],
    "chance_of_playing": [None, None, 0, 25, None],
    "news": ["", "", "Knee injury", "Knock - 25% chance", ""],
    "available": [True, True, False, False, True],
    "minutes": [3200, 900, 2000, 2500, 0],
    "starts": [36, 8, 24, 29, 0],
})


def test_nailed_starter_has_high_p_start():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.75


def test_rotation_risk_has_lower_p_start_than_nailed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[2, "p_start"] < df.loc[1, "p_start"]


def test_injured_player_is_zeroed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[3, "p_start"] == 0.0
    assert df.loc[3, "e_minutes"] == 0.0
    assert any("unavailable" in f.lower() or "injur" in f.lower() for f in df.loc[3, "flags"])


def test_doubtful_player_scaled_by_chance_of_playing():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 < df.loc[4, "p_start"] < 0.4
    assert any("25%" in f for f in df.loc[4, "flags"])


def test_new_signing_gets_price_prior_and_low_confidence():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[5, "confidence"] == "low"
    assert df.loc[5, "p_start"] > 0
    assert any("limited data" in f.lower() for f in df.loc[5, "flags"])


def test_expected_minutes_bounded_by_start_and_sub_values():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 <= df.loc[1, "e_minutes"] <= M_START + M_SUB
    assert df.loc[1, "e_minutes"] > df.loc[2, "e_minutes"]


def test_p_60_never_exceeds_p_play():
    df = minutes_model(PLAYERS, CFG)
    assert (df["p_60"] <= df["p_play"] + 1e-9).all()


def test_news_override_blended_by_weight_and_flagged():
    news = {2: {"p_start_override": 1.0, "note": "confirmed to start", "source": "example.com"}}
    base = minutes_model(PLAYERS, CFG).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, CFG, news=news).set_index("player_id")
    assert out.loc[2, "p_start"] > base
    assert any("example.com" in f for f in out.loc[2, "flags"])


def test_news_weight_zero_ignores_news():
    news = {2: {"p_start_override": 1.0, "note": "confirmed", "source": "example.com"}}
    cfg = Config(shrinkage_minutes=900, news_weight=0.0)
    base = minutes_model(PLAYERS, cfg).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, cfg, news=news).set_index("player_id").loc[2, "p_start"]
    assert abs(out - base) < 1e-9


# --- P1: beta-binomial start probability (2026-08-27 audit) ---
# The old minutes-weighted shrinkage capped p_start at ~0.85 pool-wide and pulled
# every established starter toward a mean taken over players who never played.

# 30 reserves who never played, built from the existing new-signing row so the
# concatenation below keeps identical dtypes.
ZERO_MINUTE_BENCH = pd.concat([PLAYERS.iloc[[4]]] * 30, ignore_index=True).assign(
    player_id=list(range(10, 40)),
    web_name=[f"Reserve{i}" for i in range(30)],
    team_id=9,
    price=4.5,
)


def test_ever_present_starter_is_treated_as_nailed():
    """36 of 38 starts must read as near-certain, not as a coin-flip-plus."""
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.90


def test_players_who_never_played_do_not_drag_down_the_prior():
    """The positional prior is estimated from established players only, so a pool
    padded with 30 zero-minute reserves must not move a starter's p_start."""
    alone = minutes_model(PLAYERS, CFG).set_index("player_id")
    padded = minutes_model(pd.concat([PLAYERS, ZERO_MINUTE_BENCH], ignore_index=True),
                           CFG).set_index("player_id")
    assert padded.loc[1, "p_start"] == pytest.approx(alone.loc[1, "p_start"], abs=0.02)


def test_start_prior_games_controls_how_far_the_estimate_shrinks():
    """A weaker prior leaves an established starter closer to his own start rate."""
    weak = minutes_model(PLAYERS, Config(start_prior_games=1.0)).set_index("player_id")
    strong = minutes_model(PLAYERS, Config(start_prior_games=20.0)).set_index("player_id")
    raw = 36 / 38
    assert abs(weak.loc[1, "p_start"] - raw) < abs(strong.loc[1, "p_start"] - raw)


# --- P3: season-to-date starts (2026-08-27 audit) ---

CURRENT_STARTS = pd.DataFrame({
    "player_id": [1, 2, 5],
    "gws_played": [6, 6, 6],
    "starts": [0, 6, 6],     # the nailed man benched, the rotation man nailed
    "minutes": [30.0, 540.0, 540.0],
})


def test_this_seasons_starts_override_last_seasons_reputation():
    """An ever-present who has not started a game this season is not nailed on
    any more, and the model has to be able to see that."""
    before = minutes_model(PLAYERS, CFG).set_index("player_id")
    after = minutes_model(PLAYERS, CFG, current=CURRENT_STARTS).set_index("player_id")
    assert after.loc[1, "p_start"] < before.loc[1, "p_start"]
    assert after.loc[2, "p_start"] > before.loc[2, "p_start"]


def test_a_new_signing_who_starts_stops_being_priced_off_his_transfer_fee():
    """Player 5 has no Premier League history, so his start probability came
    from his price alone — the blind spot that mispriced Tzolis."""
    before = minutes_model(PLAYERS, CFG).set_index("player_id")
    after = minutes_model(PLAYERS, CFG, current=CURRENT_STARTS).set_index("player_id")
    assert after.loc[5, "p_start"] > before.loc[5, "p_start"]
    assert not any("inferred from price" in f for f in after.loc[5, "flags"])


def test_an_unavailable_player_is_still_zeroed_whatever_his_form():
    after = minutes_model(PLAYERS, CFG, current=CURRENT_STARTS).set_index("player_id")
    assert after.loc[3, "p_start"] == 0.0


# --- Sixty minutes is not the same event as starting (2026-09-07 review) ---

def _rounds(pid, minutes_per_start):
    return pd.DataFrame([
        {"player_id": pid, "round": r, "starts": 1 if m > 0 else 0, "minutes": m,
         "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
         "saves": 0, "bonus": 0, "bps": 0, "yellow_cards": 0, "red_cards": 0,
         "own_goals": 0, "defensive_contribution": 0, "total_points": 2}
        for r, m in enumerate(minutes_per_start, start=1)
    ])


def test_p60_is_below_p_start_because_starts_do_not_all_last_the_hour():
    """`p_60 = p_start` paid a full clean sheet to every starter, including one
    who is habitually withdrawn before the hour."""
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[1, "p_60"] < df.loc[1, "p_start"]
    assert df.loc[1, "p_60"] > 0.5 * df.loc[1, "p_start"]


def test_a_player_hooked_early_every_week_reaches_sixty_less_often():
    hooked = _rounds(1, [55, 50, 58, 52, 57, 49, 54, 51])
    lasted = _rounds(1, [90, 88, 90, 90, 85, 90, 90, 90])
    early = minutes_model(PLAYERS, CFG, rounds=hooked).set_index("player_id")
    full = minutes_model(PLAYERS, CFG, rounds=lasted).set_index("player_id")
    assert early.loc[1, "p_60"] < full.loc[1, "p_60"]
    assert early.loc[1, "e_minutes"] < full.loc[1, "e_minutes"]


def test_expected_minutes_follow_the_players_own_starts_not_a_constant():
    """M_START was a flat 80 for everyone. A player who plays the full 90 every
    week and one who is always withdrawn on 55 are not the same asset."""
    full = minutes_model(PLAYERS, CFG, rounds=_rounds(1, [90] * 10)).set_index("player_id")
    assert full.loc[1, "e_minutes"] > M_START * float(full.loc[1, "p_start"])


def test_without_per_round_history_the_league_defaults_stand():
    """Pre-season there are no rounds to learn from; the model must still run."""
    df = minutes_model(PLAYERS, CFG, rounds=None).set_index("player_id")
    assert 0 < df.loc[1, "p_60"] <= df.loc[1, "p_start"]
    assert df.loc[1, "e_minutes"] > 0


def test_minutes_model_exposes_expected_minutes_given_a_start():
    """The nonlinear scoring thresholds (DC, saves, conceded) need the minutes a
    player logs WHEN HE STARTS, not the blended expectation across starting and
    not starting -- a player is 90 minutes or 0, never 73."""
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert "m_start" in df.columns
    # Nailed starter: e_minutes is dragged below m_start by the chance he sits.
    assert df.loc[1, "m_start"] > df.loc[1, "e_minutes"]
    assert 45.0 <= df.loc[1, "m_start"] <= 90.0


def test_m_start_follows_the_players_own_substitution_pattern():
    hooked = minutes_model(PLAYERS, CFG, rounds=_rounds(1, [55] * 10)).set_index("player_id")
    lasted = minutes_model(PLAYERS, CFG, rounds=_rounds(1, [90] * 10)).set_index("player_id")
    assert hooked.loc[1, "m_start"] < lasted.loc[1, "m_start"]


# --- established starter dropped from the matchday squad (2026-09-13) ------
# A player with a full established season last year (37/38 starts) whose
# CLUB has played several games this season with him recording zero minutes
# every time is not "a bit rotated" -- he has lost his place. The plain
# season-long blend cannot see this: it sums current starts over current
# games played, so one early start followed by three unused games (1/4) reads
# almost the same as even, healthy rotation, and last season's 37/38 still
# dominates the shrinkage. p_start came out at 0.88 for exactly this player
# (Bournemouth's Senesi, real GW1-4 2026/27 data) -- he was then picked as a
# starting XI defender by the optimizer.

ESTABLISHED = PLAYERS.copy()
ESTABLISHED.loc[ESTABLISHED.player_id == 1, ["minutes", "starts"]] = [3288, 37]


def test_a_trailing_absence_streak_overrides_an_established_reputation():
    """One start (round 1) then three straight unused games: last season's
    37/38 record must not be allowed to carry him as a likely starter."""
    rounds = pd.concat([_rounds(1, [90]), _rounds(1, [0, 0, 0])], ignore_index=True)
    rounds["round"] = range(1, 5)
    df = minutes_model(ESTABLISHED, CFG, rounds=rounds).set_index("player_id")
    assert df.loc[1, "p_start"] < 0.4
    assert df.loc[1, "confidence"] == "low"
    assert any("lost his place" in f.lower() or "unused" in f.lower()
              for f in df.loc[1, "flags"])


def test_a_single_rested_game_does_not_trigger_the_flag():
    """One missed game is normal rotation or a rest, not evidence of exclusion.
    Flagging on a single absence would fire on almost every squad player."""
    rounds = pd.concat([_rounds(1, [90, 90, 90]), _rounds(1, [0])], ignore_index=True)
    rounds["round"] = range(1, 5)
    df = minutes_model(ESTABLISHED, CFG, rounds=rounds).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.7
    assert not any("lost his place" in f.lower() for f in df.loc[1, "flags"])


def test_a_player_still_starting_every_current_game_is_unaffected():
    rounds = _rounds(1, [90, 88, 90, 90])
    df = minutes_model(ESTABLISHED, CFG, rounds=rounds).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.8
    assert not any("lost his place" in f.lower() for f in df.loc[1, "flags"])


def test_the_absence_streak_only_counts_the_MOST_RECENT_games():
    """Missing the start of the season and returning to nail down a place must
    not be punished for early absences that are no longer representative."""
    rounds = pd.concat([_rounds(1, [0, 0, 0]), _rounds(1, [90, 90, 90])], ignore_index=True)
    rounds["round"] = range(1, 7)
    df = minutes_model(ESTABLISHED, CFG, rounds=rounds).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.7


# --- last season's denominator must reflect AVAILABILITY (2026-09-14) -------
# `past_games` charged a flat 38 to anyone with a single minute last season, so
# a player who missed most of it injured read as "had 38 chances to start and
# took 4". Wissa (517 mins / 4 starts in 2025/26, then 4/4 starts in GW1-4
# 2026/27) came out at p_start 0.225 -- while an OTHERWISE IDENTICAL player
# with zero minutes last season routed to the price prior and got 0.791. Having
# been injured was punished 3.5x harder than never having played. 44 players
# who had started every game of 2026/27 were rated below 0.60, including Isak
# (0.31) and Odegaard (0.49).

_AVAIL_COLS = ["player_id", "web_name", "position", "team_id", "price", "status",
               "chance_of_playing", "news", "available", "minutes", "starts"]


def _avail_players(rows):
    return pd.DataFrame(rows, columns=_AVAIL_COLS)


def _current(pid, starts, games):
    return pd.DataFrame([{"player_id": pid, "gws_played": games, "starts": starts}])


RETURNEE = _avail_players([
    # injured most of last season, nailed this one
    [1, "Returnee", "FWD", 1, 6.2, "a", None, "", True, 517, 4],
    # identical, but no Premier League history at all
    [2, "Newcomer", "FWD", 2, 6.2, "a", None, "", True, 0, 0],
    # last season's ever-present, for the regression guard
    [3, "EverPresent", "FWD", 3, 7.9, "a", None, "", True, 3282, 37],
])

_CUR = pd.concat([_current(1, 4, 4), _current(2, 4, 4), _current(3, 4, 4)],
                 ignore_index=True)


def test_an_injured_season_is_not_worse_than_no_history_at_all():
    """The discontinuity that made this a bug: 517 minutes of evidence must not
    rate a player BELOW an identical player the model knows nothing about."""
    df = minutes_model(RETURNEE, CFG, current=_CUR).set_index("player_id")
    # Not parity: 4 starts from 38 IS weaker evidence than an unknown player at
    # the same price, so rating him somewhat lower is correct. What must not
    # survive is the CLIFF -- 0.225 against 0.791, a factor of 3.5 -- so the bar
    # is that history costs him a fraction, not a multiple.
    assert df.loc[1, "p_start"] >= df.loc[2, "p_start"] * 0.8


def test_a_returning_starter_is_rated_as_a_starter():
    """Four starts from four games, fully available, is a starter -- last
    season's injury must not cap him near the rotation band."""
    df = minutes_model(RETURNEE, CFG, current=_CUR).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.6


def test_an_ever_present_is_essentially_unchanged():
    """The fix must target the injured cohort and leave established players
    alone, or it is just a global loosening of the shrinkage."""
    df = minutes_model(RETURNEE, CFG, current=_CUR).set_index("player_id")
    assert df.loc[3, "p_start"] > 0.90


def test_p_start_stays_a_probability():
    from fpl.model.minutes import TEAM_GAMES
    huge = _avail_players([[1, "Iron", "DEF", 1, 5.0, "a", None, "", True,
                           TEAM_GAMES * 95, 38]])
    df = minutes_model(huge, CFG).set_index("player_id")
    assert 0.0 <= df.loc[1, "p_start"] <= 1.0


# --- this season decides; last season is only a prior (2026-09-14) ---------
# Pooling both seasons into one ratio gave last season a 38-game denominator
# against this season's 4, so current evidence was ~9% of the signal and a
# player's ACTUAL current role could barely move him. Roles change between
# seasons -- transfers, new managers, new signings -- so recent starts are the
# better evidence, and last season belongs in the prior.

DROPPED = _avail_players([
    # last season's regular who has not started a game this season
    [1, "Dropped", "MID", 1, 5.0, "a", None, "", True, 1560, 19],
    # last season's ever-present, still starting
    [2, "Nailed", "MID", 2, 7.9, "a", None, "", True, 3282, 37],
    # thin last season through injury, starting every game now
    [3, "Returned", "MID", 3, 6.2, "a", None, "", True, 517, 4],
])


def test_starting_every_game_this_season_reads_as_a_starter():
    """The user's rule: 4 starts from 4 games means he is most likely starting,
    whatever last season looked like."""
    cur = pd.concat([_current(1, 0, 4), _current(2, 4, 4), _current(3, 4, 4)],
                    ignore_index=True)
    df = minutes_model(DROPPED, CFG, current=cur).set_index("player_id")
    assert df.loc[3, "p_start"] > 0.65
    assert df.loc[2, "p_start"] > 0.85


def test_not_starting_this_season_outweighs_last_seasons_reputation():
    """The mirror of the same rule, and the case the availability estimator got
    backwards: 0 starts from 4 games is not a nailed starter."""
    cur = pd.concat([_current(1, 0, 4), _current(2, 4, 4), _current(3, 4, 4)],
                    ignore_index=True)
    df = minutes_model(DROPPED, CFG, current=cur).set_index("player_id")
    assert df.loc[1, "p_start"] < 0.40
    assert df.loc[1, "p_start"] < df.loc[3, "p_start"]


def test_before_a_ball_is_kicked_last_season_is_all_there_is():
    """With no current-season games the prior must still separate an
    ever-present from a squad player."""
    df = minutes_model(DROPPED, CFG).set_index("player_id")
    assert df.loc[2, "p_start"] > df.loc[1, "p_start"] > df.loc[3, "p_start"]


def test_a_thin_last_season_is_pulled_toward_the_positional_average():
    """4 starts in 517 minutes is a small sample, so it must not be taken at
    face value as a 10% start rate -- that is what buried the injured cohort."""
    df = minutes_model(DROPPED, CFG).set_index("player_id")
    assert df.loc[3, "p_start"] > 4.0 / 38.0 * 1.5


# --- B9: availability caps every route onto the pitch (2026-09-17 audit) ---

def test_an_unavailable_player_has_no_route_onto_the_pitch():
    """p_start was zeroed and p_play then rebuilt from the generic cameo rule,
    leaving a ruled-out player a 35% chance of appearing. Deterministic xP read
    e_minutes and returned zero; the simulator read p_play and put him on."""
    m = minutes_model(PLAYERS, CFG).set_index("player_id").loc[3]
    assert m.p_start == 0.0
    assert m.p_play == 0.0
    assert m.p_60 == 0.0
    assert m.e_minutes == 0.0


def test_a_doubtful_player_loses_the_cameo_too():
    """A 25% chance of playing is 25% of BOTH ways he could play, not a quarter
    of a start plus a full cameo."""
    fit = PLAYERS.copy()
    fit.loc[fit.player_id == 4, "status"] = "a"
    fit.loc[fit.player_id == 4, "chance_of_playing"] = None

    healthy = minutes_model(fit, CFG).set_index("player_id").loc[4]
    doubt = minutes_model(PLAYERS, CFG).set_index("player_id").loc[4]
    assert doubt.p_start == pytest.approx(healthy.p_start * 0.25)
    assert doubt.p_play == pytest.approx(healthy.p_play * 0.25)


def test_a_start_rate_does_not_depend_on_how_matches_are_packed_into_gameweeks():
    """Two starts in three matches is the same evidence whether the three
    matches fell in three gameweeks or in two with a double.

    `starts` sums fixture ROWS while the denominator counted distinct ROUNDS,
    so the double-gameweek player got the same two starts over one fewer
    opportunity and read as the more nailed of the two."""
    from fpl.data.normalize import history_current_frame

    spread = {2: {"history": [
        {"round": 1, "starts": 1, "minutes": 90},
        {"round": 2, "starts": 1, "minutes": 90},
        {"round": 3, "starts": 0, "minutes": 5},
    ]}}
    doubled = {2: {"history": [
        {"round": 1, "starts": 1, "minutes": 90},
        {"round": 2, "starts": 1, "minutes": 90},
        {"round": 2, "starts": 0, "minutes": 5},
    ]}}

    a = minutes_model(PLAYERS, CFG,
                      current=history_current_frame(spread, before_event=4))
    b = minutes_model(PLAYERS, CFG,
                      current=history_current_frame(doubled, before_event=4))
    assert (a.set_index("player_id").loc[2, "p_start"]
            == pytest.approx(b.set_index("player_id").loc[2, "p_start"]))
