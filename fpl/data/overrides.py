"""Declarative team-news minutes overrides.

`fpl.model.minutes.minutes_model` blends a `p_start_override` into its own prior
at `news.weight`, but nothing ever populated that dict -- so corrections the
model structurally cannot see had to be hand-applied in throwaway scripts.

The gap they exist to close: the minutes model is per-player, not a team-level
allocation. It sets an injured player's own p_start to 0 but never redistributes
those minutes to whoever deputises, so a stand-in's start probability stays
pinned to his own thin history no matter who is out ahead of him.

Every override must carry a `source` and should carry an `until_gw`, so a stale
correction expires instead of quietly outliving the news that justified it.

It should also carry `checked_at`: the date the news was last verified. An
`until_gw` of 6 keeps an override alive for a month of gameweeks, which is far
longer than a fitness report stays true -- `news.max_age_hours` from config.yaml
is what flags one that has outlived its evidence.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import yaml

REQUIRED = ("player_id", "p_start_override", "source")


def _as_datetime(value) -> datetime | None:
    """Parse a YAML date/datetime/ISO string as UTC, or None if unusable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def override_age_hours(checked_at, now: datetime | None = None) -> float | None:
    """Hours since an override was last verified, or None if it never was."""
    when = _as_datetime(checked_at)
    if when is None:
        return None
    now = now or datetime.now(timezone.utc)
    now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return (now - when).total_seconds() / 3600.0


def load_overrides(path: Path, gw: int, max_age_hours: float | None = None,
                   now: datetime | None = None) -> dict[int, dict]:
    """Return {player_id: override} for overrides still active at `gw`.

    A missing or empty file is normal and yields {}. `until_gw` is inclusive:
    an override written for GW1 no longer applies at GW2.

    `max_age_hours` (config `news.max_age_hours`) marks an override `stale`
    when its `checked_at` is older than that, or missing entirely. Stale
    overrides are still applied -- silently dropping a correction the model
    cannot make for itself would be worse -- but the runner says so loudly, so
    the choice to keep trusting month-old team news is made deliberately.
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("overrides") or []

    out: dict[int, dict] = {}
    for entry in entries:
        for key in REQUIRED:
            if entry.get(key) is None:
                raise ValueError(
                    f"override {entry!r} is missing required field {key!r} -- "
                    f"every override needs a player_id, a probability and a "
                    f"citable source."
                )
        prob = float(entry["p_start_override"])
        if not 0.0 <= prob <= 1.0:
            raise ValueError(
                f"p_start_override must be a probability in 0..1, got {prob} "
                f"for player_id {entry['player_id']}"
            )
        until = entry.get("until_gw")
        if until is not None and int(gw) > int(until):
            continue
        checked_at = entry.get("checked_at")
        age = override_age_hours(checked_at, now)
        stale = max_age_hours is not None and (age is None or age > float(max_age_hours))
        out[int(entry["player_id"])] = {
            "p_start_override": prob,
            "note": str(entry.get("note", "")),
            "source": str(entry["source"]),
            "until_gw": until,
            "checked_at": None if checked_at is None else str(checked_at),
            "age_hours": age,
            "stale": stale,
        }
    return out
