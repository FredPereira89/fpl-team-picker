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
"""
from pathlib import Path
import yaml

REQUIRED = ("player_id", "p_start_override", "source")


def load_overrides(path: Path, gw: int) -> dict[int, dict]:
    """Return {player_id: override} for overrides still active at `gw`.

    A missing or empty file is normal and yields {}. `until_gw` is inclusive:
    an override written for GW1 no longer applies at GW2.
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
        out[int(entry["player_id"])] = {
            "p_start_override": prob,
            "note": str(entry.get("note", "")),
            "source": str(entry["source"]),
            "until_gw": until,
        }
    return out
