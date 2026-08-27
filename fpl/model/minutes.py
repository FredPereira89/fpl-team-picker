"""Minutes model: probability of starting, appearing, and reaching 60 minutes."""
import pandas as pd

M_START = 80.0     # typical minutes when a player starts
M_SUB = 20.0       # typical minutes when a player comes off the bench
P_SUB_APPEAR = 0.35  # chance a non-starter appears at all
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


def minutes_model(players: pd.DataFrame, cfg, news: dict[int, dict] | None = None) -> pd.DataFrame:
    news = news or {}
    # Starts are binomial over a fixed 38-game season, so the natural shrinkage
    # is a beta-binomial measured in GAMES, not the minutes-weighted blend used
    # for per-90 rates. Shrinking on minutes (k=900) mixed the two scales and
    # left an ever-present starter at ~0.85 -- the model could not express
    # "nailed on", which is the distinction transfers and captaincy turn on.
    k_games = float(cfg.start_prior_games)
    pos_prior, overall_prior = start_priors(players)

    rows = []
    for _, p in players.iterrows():
        flags: list[str] = []
        confidence = "high"
        minutes = float(p["minutes"])

        if minutes <= 0:
            p_start = _price_prior(float(p["price"]), p["position"])
            confidence = "low"
            flags.append(
                f"Limited data: no Premier League minutes on record — "
                f"start probability inferred from price (£{p['price']}m)"
            )
        else:
            prior = float(pos_prior.get(p["position"], overall_prior))
            p_start = (float(p["starts"]) + k_games * prior) / (TEAM_GAMES + k_games)
            if minutes < ESTABLISHED_MINUTES:
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
        e_minutes = p_start * M_START + (p_play - p_start) * M_SUB if p_start > 0 else 0.0
        p_60 = p_start  # reaching 60 minutes effectively requires starting

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
