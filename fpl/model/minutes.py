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


# How fast a stale override fades. At its freshness limit an override is
# applied in full; every further limit's worth of age halves its weight, so a
# note checked a week ago against a 48-hour budget is worth about a tenth.
STALE_HALF_LIFE_BUDGETS = 1.0


def override_trust(override: dict, cfg) -> float:
    """How much of a manual override's weight it still deserves, 0..1.

    Stale overrides used to be applied at full configured weight forever,
    with a warning nobody could act on from inside the model. A note that was
    right on the day was quietly setting minutes a month later. It now decays
    toward the model past its freshness budget rather than being dropped --
    dropping it would be worse, the correction is usually still directionally
    right -- and the flag says so.
    """
    if not override.get("stale"):
        return 1.0
    budget = float(getattr(cfg, "news_max_age_hours", 0) or 0)
    age = override.get("age_hours")
    if budget <= 0 or age is None:
        return 0.5
    over = max(0.0, float(age) - budget) / budget
    return float(0.5 ** (over / STALE_HALF_LIFE_BUDGETS))


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
        # MATCHES his club has played since he joined it -- not gameweeks, and
        # not games he featured in. `now_starts` sums fixture rows, so the
        # denominator has to count them too: counting distinct rounds gave a
        # double-gameweek player two starts out of one opportunity, and a
        # rotation risk then read as nailed on for every week after.
        now_games = (float(seen.get("matches_played", seen["gws_played"]))
                     if seen is not None else 0.0)
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
        # How many games of evidence sit behind that number. A price prior and
        # an ever-present's thirty starts can produce the same p_start; only
        # this says which one it is. Not currently consumed by the simulator
        # -- a single gameweek's appearance event cannot be "widened" by a
        # per-scenario posterior draw, only biased if done carelessly (see
        # model.simulate's history) -- kept as a labelled confidence signal
        # for a future extension that reuses the SAME draw across a
        # multi-week decision, where the correlation would be real.
        start_evidence = now_games + PRIOR_WEIGHT_GAMES
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
        # Team news blends into the FULLY-FIT start probability, and availability
        # is applied once, afterwards. Blending it into the already-capped value
        # let a confident "he starts" note lift a 25%-available player above
        # 0.25 -- and then p_play was capped at availability, leaving the
        # impossible state p_start > p_play, which the simulator resolved by
        # starting him far more often than the doubt allowed.
        override = news.get(int(p["player_id"]))
        if override and cfg.news_weight > 0 and availability > 0:
            w = float(cfg.news_weight) * override_trust(override, cfg)
            p_start = (1 - w) * p_start + w * float(override["p_start_override"])
            note = f"Team news: {override['note']} (source: {override['source']})"
            if override.get("stale"):
                note += (f" — STALE, last checked {override.get('checked_at') or 'never'}; "
                         f"applied at {w:.2f} weight and fading")
            flags.append(note)

        fit_start = float(min(1.0, max(0.0, p_start)))
        p_start = availability * fit_start
        # The cameo branch is capped by the SAME availability: a 25% doubt takes
        # a quarter of the cameo as well as of the start.
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
            # Effective games behind p_start, for a Beta posterior in the
            # simulation: alpha = p * n, beta = (1 - p) * n.
            "start_evidence": float(start_evidence),
            "confidence": confidence,
            "flags": flags,
        })
    return reconcile_team_starts(pd.DataFrame(rows).reset_index(drop=True), players)


# A side starts eleven. A team whose modelled start probabilities sum past
# this is promising more starts than exist, and every one of its players is
# over-projected together -- exactly the correlated error a stack then buys.
STARTERS_PER_TEAM = 11.0


def reconcile_team_starts(minutes: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Scale a team's start probabilities DOWN so they promise at most eleven.

    Each player's p_start is estimated on his own, so a squad in which every
    fringe player has a plausible case could sum to fourteen starters. The
    simulation then fielded more than eleven and the projection paid for
    them. Scaling the team's starts to eleven when they exceed it keeps every
    ordering intact and removes the shared over-promise; p_play, p_60 and
    e_minutes follow the start branch down.

    Deliberately never scales UP: a team summing to nine is more often a
    team with genuine uncertainty over two places than one whose regulars
    are under-rated, and inflating everyone would hand starts to players
    the evidence does not support. The team-level shortfall is a known
    residual of R5 (event-specific, depth-chart minutes), not fixed here.
    """
    if len(minutes) == 0 or "team_id" not in players.columns:
        return minutes
    out = minutes.copy()
    team_of = dict(zip(players["player_id"].astype(int), players["team_id"].astype(int)))
    teams = out["player_id"].astype(int).map(team_of)
    totals = out["p_start"].groupby(teams).sum()
    factor = teams.map(lambda t: min(1.0, STARTERS_PER_TEAM / totals[t])
                       if t in totals.index and totals[t] > 0 else 1.0).astype(float)
    if (factor >= 1.0 - 1e-12).all():
        return out
    scaled = factor < 1.0
    for col in ("p_start", "p_play", "p_60", "e_minutes"):
        if col in out.columns:
            out.loc[scaled, col] = out.loc[scaled, col] * factor[scaled]
    for i in out.index[scaled]:
        out.at[i, "flags"] = list(out.at[i, "flags"]) + [
            f"Start rate scaled by {factor[i]:.2f}: his side's modelled starters "
            f"summed past eleven"]
    return out
