"""Minutes model: probability of starting, appearing, and reaching 60 minutes."""
import pandas as pd

M_START = 80.0     # typical minutes when a player starts
M_SUB = 20.0       # typical minutes when a player comes off the bench
P_SUB_APPEAR = 0.35  # chance a non-starter appears at all
TEAM_GAMES = 38
UNAVAILABLE = {"i", "s", "u"}
DOUBTFUL = "d"


def _price_prior(price: float, position: str) -> float:
    """Prior p_start for a player with no PL history, from FPL's own pricing signal."""
    floors = {"GKP": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
    floor = floors.get(position, 4.5)
    return float(min(0.85, max(0.15, (price - floor) / 6.0 + 0.25)))


def minutes_model(players: pd.DataFrame, cfg, news: dict[int, dict] | None = None,
                   games_played: int | None = None) -> pd.DataFrame:
    """`players["starts"]`/`minutes` are FPL's own season-cumulative fields.

    Before a season's first ball is kicked they still hold the *prior*
    season's full total, so dividing by a full TEAM_GAMES is correct pre-
    season. Once the tracked season is under way those same fields reset and
    only accumulate the *current* season's results, so the rate must be
    taken over games actually played so far -- dividing by a fixed 38 there
    would crush a nailed GW1 starter's rate to ~1/38 and wreck in-season
    p_start. Callers must pass `games_played` (games completed so far this
    season) once any have been played; omitting it defaults to the
    full-season/pre-season divisor.
    """
    news = news or {}
    k = float(cfg.shrinkage_minutes)
    in_season = bool(games_played)
    divisor = float(games_played) if in_season else TEAM_GAMES
    pos_mean = (players["starts"] / divisor).mean()
    sample_period = "this season" if in_season else "last season"

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
            raw = float(p["starts"]) / divisor
            p_start = (minutes * raw + k * pos_mean) / (minutes + k)
            if minutes < 900:
                confidence = "medium"
                flags.append(f"Small sample: {int(minutes)} minutes {sample_period}")

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
