"""Expected bonus points.

Models BONUS ONLY. Defensive Contribution threshold points are direct
scoring and are handled in model/xp.py — the same raw actions feed both
mechanisms, but they must not be awarded twice.
"""
import pandas as pd

MAX_BONUS_PER_MATCH = 3.0
FIXTURE_SENSITIVITY = 0.5  # bonus is less fixture-dependent than goals


def expected_bonus(rates: pd.DataFrame, minutes: pd.DataFrame,
                   att_mult: float = 1.0) -> pd.Series:
    df = rates[["player_id", "bonus90"]].merge(
        minutes[["player_id", "e_minutes"]], on="player_id", how="left"
    ).fillna({"e_minutes": 0.0})

    scale = 1.0 + (att_mult - 1.0) * FIXTURE_SENSITIVITY
    per_match = df["bonus90"] * (df["e_minutes"] / 90.0) * scale
    per_match = per_match.clip(lower=0.0, upper=MAX_BONUS_PER_MATCH)

    out = pd.Series(per_match.values, index=df["player_id"].astype(int))
    out.index.name = "player_id"
    return out
