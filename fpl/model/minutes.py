"""Minutes model: probability of starting, appearing, and reaching 60 minutes."""
import pandas as pd

M_START = 80.0     # typical minutes when a player starts
M_SUB = 20.0       # typical minutes when a player comes off the bench
P_SUB_APPEAR = 0.35  # chance a non-starter appears at all
# Share of starts that last the hour. Not 1.0: a start is worth two appearance
# points and a clean sheet only if it reaches 60 minutes, and equating the two
# -- which this model did, with `p_60 = p_start` -- silently pays a full clean
# sheet to every player who has ever been hooked at half time.
P60_GIVEN_START = 0.88
# Starts of evidence before a player's own 60-minute rate outweighs the league
# one. Deliberately small: substitution patterns are a manager's habit and show
# up fast, unlike scoring rates.
P60_PRIOR_STARTS = 5.0
TEAM_GAMES = 38
UNAVAILABLE = {"i", "s", "u"}
DOUBTFUL = "d"
# A player needs roughly a third of a season before his own start rate is worth
# more than the positional prior. Below this he still counts as a small sample
# for the confidence flag, and he is excluded from the prior he is shrunk toward.
ESTABLISHED_MINUTES = 900.0
FALLBACK_PRIOR = 0.35  # used only when no player in the pool has any minutes


def _price_prior(price: float, position: str) -> float:
    """Prior p_start for a player with no PL history, from FPL's own pricing signal."""
    floors = {"GKP": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
    floor = floors.get(position, 4.5)
    return float(min(0.85, max(0.15, (price - floor) / 6.0 + 0.25)))


def start_priors(players: pd.DataFrame) -> tuple[pd.Series, float]:
    """Positional start rates, estimated from established players only.

    Averaging `starts / 38` over the whole pool includes the several hundred
    players who never featured, which drags the prior toward zero and, through
    the shrinkage below, caps what any starter can be assigned. Only players
    past ESTABLISHED_MINUTES carry information about what a starting role
    looks like, so only they define the prior.
    """
    est = players[players["minutes"].astype(float) >= ESTABLISHED_MINUTES]
    if len(est) == 0:
        est = players[players["minutes"].astype(float) > 0]
    if len(est) == 0:
        return pd.Series(dtype="float64"), FALLBACK_PRIOR
    rates = est["starts"].astype(float) / TEAM_GAMES
    return rates.groupby(est["position"]).mean(), float(rates.mean())


# How many games of THIS season last season's reputation is worth. Small on
# purpose: a player's role is remade every summer, so four current gameweeks
# should outweigh it, not merely nudge it. At 2.0 a player who has started all
# four games so far reads ~0.72 off a poor prior and ~0.98 off a strong one,
# while one who has started none of them falls to ~0.17 whatever he did last
# year. This is the knob to calibrate against scored gameweeks.
PRIOR_WEIGHT_GAMES = 2.0

MIN_START_MINUTES = 45.0  # a "start" row with fewer minutes than this is an error

# Trailing team-games with zero minutes before an established reputation is
# treated as stale. One missed game is normal rotation or a rest; several
# straight unused games despite being nominally "available" is a manager
# losing his place, and last season's start count must not be allowed to
# carry him past it -- a season-long blend cannot see this at all, because
# one early start followed by three unused games (1/4) looks almost the same
# as healthy even rotation in a cumulative count. Only the RECENCY of the
# zeros -- all three being the most recent games, not spread through the
# season -- distinguishes "dropped" from "rotated", found 2026-09-13 on real
# GW1-4 data: an established defender (37/38 starts last season) who started
# once then recorded zero minutes for three straight games still came out at
# p_start=0.88 and was picked in a starting XI.
ABSENCE_STREAK_GRACE = 1
ABSENCE_STREAK_DECAY = 0.5  # multiplies p_start per trailing zero beyond the grace


def trailing_absence_streaks(rounds: pd.DataFrame | None) -> dict[int, int]:
    """{player_id: how many of his own MOST RECENT team-games he had zero
    minutes in}, counting back from the latest game until the first one he
    actually played.
    """
    if rounds is None or len(rounds) == 0:
        return {}
    out: dict[int, int] = {}
    for pid, g in rounds.groupby(rounds["player_id"].astype(int)):
        mins = g.sort_values("round")["minutes"].astype(float).to_numpy()
        streak = 0
        for m in mins[::-1]:
            if m > 0:
                break
            streak += 1
        out[int(pid)] = streak
    return out


def start_profiles(rounds: pd.DataFrame | None) -> tuple[dict[int, dict], float, float]:
    """Per player: how often a start lasts the hour, and how long it lasts.

    Returns ({player_id: {"starts", "p60", "m_start"}}, league p60, league
    minutes-per-start). Both were fixed constants before: every start was
    assumed to reach 60 minutes and to last exactly 80. That is wrong in both
    directions for the players it matters most for -- a striker routinely
    withdrawn on 65 minutes, a defender rotated at 60 -- and those minutes feed
    appearance points, clean sheets, defensive contributions and autosubs.
    """
    if rounds is None or len(rounds) == 0:
        return {}, P60_GIVEN_START, M_START
    started = rounds[rounds["starts"].astype(float) > 0]
    if len(started) == 0:
        return {}, P60_GIVEN_START, M_START

    mins = started["minutes"].astype(float)
    league_p60 = float((mins >= 60).mean())
    league_m = float(mins.mean())
    profiles: dict[int, dict] = {}
    for pid, g in started.groupby(started["player_id"].astype(int)):
        m = g["minutes"].astype(float)
        profiles[int(pid)] = {
            "starts": float(len(g)),
            "p60": float((m >= 60).mean()),
            "m_start": float(m.mean()),
        }
    return profiles, league_p60, league_m


def minutes_model(players: pd.DataFrame, cfg, news: dict[int, dict] | None = None,
                  current: pd.DataFrame | None = None,
                  rounds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Start probability, appearance probability and expected minutes.

    `current` is this season's `history_current_frame`: starts and gameweeks on
    record so far. Season-to-date starts are counted as ordinary binomial
    evidence alongside last season's, which is what lets a summer signing stop
    being priced off his transfer fee, and an ever-present who has lost his
    place stop reading as nailed on.
    """
    news = news or {}
    # Starts are binomial over a fixed 38-game season, so the natural shrinkage
    # is a beta-binomial measured in GAMES, not the minutes-weighted blend used
    # for per-90 rates. Shrinking on minutes (k=900) mixed the two scales and
    # left an ever-present starter at ~0.85 -- the model could not express
    # "nailed on", which is the distinction transfers and captaincy turn on.
    k_games = float(cfg.start_prior_games)
    pos_prior, overall_prior = start_priors(players)
    now = ({int(r["player_id"]): r for _, r in current.iterrows()}
           if current is not None and len(current) else {})
    profiles, league_p60, league_m_start = start_profiles(rounds)
    absence_streaks = trailing_absence_streaks(rounds)

    rows = []
    for _, p in players.iterrows():
        flags: list[str] = []
        confidence = "high"
        minutes = float(p["minutes"])
        seen = now.get(int(p["player_id"]))
        # Games his club has played since he joined it -- not games he featured
        # in -- which is the number of chances to start he has actually had.
        now_games = float(seen["gws_played"]) if seen is not None else 0.0
        now_starts = float(seen["starts"]) if seen is not None else 0.0
        has_past = minutes > 0

        # Last season is a PRIOR, not evidence on equal footing. Pooling both
        # seasons into one ratio gave it a 38-game denominator against this
        # season's handful, so a player's actual current role was ~9% of the
        # signal: Wissa, who had started every game of 2026/27, was held at
        # p_start 0.225 by an injury-shortened 2025/26, and Hughes, who had
        # started none of them, sat at 0.79 on last season's reputation. Roles
        # do not carry over -- transfers, new managers and new signings remake
        # them every summer -- so THIS season's starts are the evidence and
        # last season only says where to begin.
        #
        # For a player with no Premier League history, price IS the prior --
        # it is FPL's own estimate of his role.
        pos = float(pos_prior.get(p["position"], overall_prior))
        if has_past:
            # Last season shrunk toward the positional average over its own 38
            # games, so 36-of-38 still reads as nailed while a handful of starts
            # does not read as a hard 10% -- 4 starts in an injury-shortened 517
            # minutes is mostly absence, and taking it literally is what buried
            # the returning-from-injury cohort.
            prior = (float(p["starts"]) + k_games * pos) / (TEAM_GAMES + k_games)
        else:
            prior = _price_prior(float(p["price"]), p["position"])

        # The prior is worth PRIOR_WEIGHT_GAMES games of this season, so a few
        # current gameweeks genuinely outweigh last season's reputation rather
        # than nudging it. With no current games this collapses to the prior.
        p_start = ((now_starts + PRIOR_WEIGHT_GAMES * prior)
                   / (now_games + PRIOR_WEIGHT_GAMES))
        if not has_past and now_games <= 0:
            confidence = "low"
            flags.append(
                f"Limited data: no Premier League minutes on record — "
                f"start probability inferred from price (£{p['price']}m)"
            )
        elif not has_past:
            confidence = "medium" if now_games >= 3 else "low"
            flags.append(
                f"No Premier League history — rated on {int(now_starts)} start(s) "
                f"in {int(now_games)} gameweek(s) this season"
            )
        elif minutes < ESTABLISHED_MINUTES:
            confidence = "medium"
            flags.append(f"Small sample: {int(minutes)} minutes last season")

        streak = absence_streaks.get(int(p["player_id"]), 0)
        if has_past and streak > ABSENCE_STREAK_GRACE:
            p_start *= ABSENCE_STREAK_DECAY ** (streak - ABSENCE_STREAK_GRACE)
            confidence = "low"
            flags.append(
                f"Established starter last season, but unused in his last "
                f"{streak} team games — may have lost his place; verify "
                f"team news before trusting his projection."
            )

        # Availability is a CAP on every route onto the pitch, not a discount on
        # one of them. Zeroing p_start alone left the generic cameo rule below
        # to hand a ruled-out player a 35% chance of appearing: deterministic xP
        # read e_minutes and returned zero, while the simulator read p_play and
        # put him on -- so the two disagreed exactly where the shared rank
        # scenarios are most sensitive to a phantom appearance.
        availability = 1.0
        status = str(p["status"])
        if status in UNAVAILABLE:
            availability = 0.0
            note = str(p["news"]).strip() or "unavailable"
            flags.append(f"Unavailable ({status}): {note}")
        elif status == DOUBTFUL:
            chance = p["chance_of_playing"]
            pct = 50.0 if pd.isna(chance) else float(chance)
            availability = pct / 100.0
            confidence = "low"
            note = str(p["news"]).strip()
            flags.append(f"Doubtful: {int(pct)}% chance of playing" + (f" — {note}" if note else ""))
        p_start *= availability

        override = news.get(int(p["player_id"]))
        if override and cfg.news_weight > 0 and p_start > 0:
            w = float(cfg.news_weight)
            p_start = (1 - w) * p_start + w * float(override["p_start_override"])
            flags.append(f"Team news: {override['note']} (source: {override['source']})")

        p_start = float(min(1.0, max(0.0, p_start)))
        # The chance he would start if fully fit, recovered from the capped
        # value so the cameo branch can be capped by the SAME availability --
        # a 25% doubt takes a quarter of the cameo as well as of the start.
        fit_start = min(1.0, p_start / availability) if availability > 0 else 0.0
        p_play = availability * (fit_start + (1.0 - fit_start) * P_SUB_APPEAR)

        # Reaching 60 minutes needs a start AND the hour: a player who starts
        # every week but is routinely withdrawn on 55 is not a clean-sheet
        # asset, and treating p_60 as p_start said he was.
        prof = profiles.get(int(p["player_id"]))
        k = P60_PRIOR_STARTS
        if prof:
            n = prof["starts"]
            p60_given_start = (prof["p60"] * n + league_p60 * k) / (n + k)
            m_start = (prof["m_start"] * n + league_m_start * k) / (n + k)
        else:
            p60_given_start, m_start = league_p60, league_m_start
        p_60 = p_start * p60_given_start
        # No `if p_start > 0` guard: availability already zeroes an unavailable
        # player through p_play, and a genuine cameo-only player does log
        # minutes. The old guard made e_minutes disagree with p_play.
        e_minutes = p_start * m_start + (p_play - p_start) * M_SUB

        rows.append({
            "player_id": int(p["player_id"]),
            "p_start": p_start,
            "p_play": p_play,
            "p_60": p_60,
            "e_minutes": e_minutes,
            # Minutes logged WHEN HE STARTS. `e_minutes` blends starting and
            # not starting, which is the right input for the linear terms
            # (goals, assists, cards) and the wrong one for every threshold:
            # a match is 90 minutes or 0, never the average of the two.
            "m_start": m_start,
            "confidence": confidence,
            "flags": flags,
        })
    return pd.DataFrame(rows).reset_index(drop=True)
