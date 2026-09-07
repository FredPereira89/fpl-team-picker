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


MIN_START_MINUTES = 45.0  # a "start" row with fewer minutes than this is an error


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
        past_games = float(TEAM_GAMES) if minutes > 0 else 0.0

        # For a player with no Premier League history, price IS the prior --
        # it is FPL's own estimate of his role. Season-to-date starts then
        # update it like any other evidence, instead of being ignored in favour
        # of a positional average he has no business being shrunk toward.
        prior = (float(pos_prior.get(p["position"], overall_prior)) if past_games > 0
                 else _price_prior(float(p["price"]), p["position"]))
        if past_games + now_games <= 0:
            p_start = _price_prior(float(p["price"]), p["position"])
            confidence = "low"
            flags.append(
                f"Limited data: no Premier League minutes on record — "
                f"start probability inferred from price (£{p['price']}m)"
            )
        else:
            p_start = ((float(p["starts"]) + now_starts + k_games * prior)
                       / (past_games + now_games + k_games))
            if minutes <= 0:
                confidence = "medium" if now_games >= 3 else "low"
                flags.append(
                    f"No Premier League history — rated on {int(now_starts)} start(s) "
                    f"in {int(now_games)} gameweek(s) this season"
                )
            elif minutes < ESTABLISHED_MINUTES:
                confidence = "medium"
                flags.append(f"Small sample: {int(minutes)} minutes last season")

        status = str(p["status"])
        if status in UNAVAILABLE:
            p_start = 0.0
            note = str(p["news"]).strip() or "unavailable"
            flags.append(f"Unavailable ({status}): {note}")
        elif status == DOUBTFUL:
            chance = p["chance_of_playing"]
            pct = 50.0 if pd.isna(chance) else float(chance)
            p_start *= pct / 100.0
            confidence = "low"
            note = str(p["news"]).strip()
            flags.append(f"Doubtful: {int(pct)}% chance of playing" + (f" — {note}" if note else ""))

        override = news.get(int(p["player_id"]))
        if override and cfg.news_weight > 0 and p_start > 0:
            w = float(cfg.news_weight)
            p_start = (1 - w) * p_start + w * float(override["p_start_override"])
            flags.append(f"Team news: {override['note']} (source: {override['source']})")

        p_start = float(min(1.0, max(0.0, p_start)))
        p_play = p_start + (1 - p_start) * P_SUB_APPEAR

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
        e_minutes = p_start * m_start + (p_play - p_start) * M_SUB if p_start > 0 else 0.0

        rows.append({
            "player_id": int(p["player_id"]),
            "p_start": p_start,
            "p_play": p_play,
            "p_60": p_60,
            "e_minutes": e_minutes,
            "confidence": confidence,
            "flags": flags,
        })
    return pd.DataFrame(rows).reset_index(drop=True)
