"""Which chips are legal to play in a given gameweek.

2026/27 gives every manager TWO of each chip: one usable in GW1-19, one in
GW20-38. An unused first-half chip EXPIRES at the GW19 deadline rather than
rolling over, so the second half always starts with exactly one of each
however the first half went.

`State.chips_used` -- a unique list of names -- cannot express any of that.
Once any Wildcard had been played it suppressed the second-half copy for the
rest of the season, which is four chips and easily twenty points. The dated
`chip_events` record can, so it is the only thing this module reads.

Responsibilities are split three ways and deliberately do not overlap: this
module says what is LEGAL, `optimize.chips` says whether a legal chip is worth
playing NOW, and `optimize.actions` says what the chip actually DOES to the
squad.
"""
from .state import canonical_chip

CHIPS = ("wildcard", "freehit", "benchboost", "triplecaptain")
# Last gameweek of the first chip window. The second set unlocks the week after.
FIRST_HALF_LAST = 19
# Wildcard and Free Hit are disabled in an entry's opening gameweek: transfers
# before that deadline are already unlimited, so neither chip can buy anything.
NO_CHIP_IN_OPENING_GW = ("wildcard", "freehit")
# Gameweeks that must separate two Free Hits. 1 means "not in consecutive
# gameweeks": a Free Hit in GW19 blocks GW20 but not GW21. Set to 0 to drop the
# restriction -- it is the one rule here taken from the audit's reading of the
# FPL FAQ rather than from observed behaviour.
FREE_HIT_MIN_GAP = 1


def chip_window(event: int) -> int:
    """1 for the first-half set of chips, 2 for the second."""
    return 1 if int(event) <= FIRST_HALF_LAST else 2


def _played(chip_events) -> list[tuple[str, int | None]]:
    """(chip, event) for every recorded use, names canonicalised."""
    out = []
    for record in chip_events or []:
        name = canonical_chip(record.get("chip"))
        if name is None:
            continue
        event = record.get("event")
        out.append((name, None if event is None else int(event)))
    return out


def chip_blocked_reason(chip, event: int, chip_events,
                        first_event: int = 1) -> str | None:
    """Why `chip` cannot be played in `event`, or None if it can.

    Returned as prose rather than a bool because the weekly report has to tell
    the user WHICH rule stopped a chip the squad otherwise qualified for --
    "already used" and "not until GW20" are different pieces of news.
    """
    name = canonical_chip(chip)
    if name is None:
        return None
    event = int(event)
    window = chip_window(event)
    played = _played(chip_events)

    if name in NO_CHIP_IN_OPENING_GW and event <= int(first_event):
        return (f"a {name} cannot be played in GW{event}, your opening gameweek "
                f"— transfers before that deadline are already unlimited")

    for other, when in played:
        if other != name:
            continue
        # An undated record predates chip_events and in practice was played
        # early, so it is charged to the first window. See the module docstring.
        used_window = 1 if when is None else chip_window(when)
        if used_window != window:
            continue
        where = "earlier this season" if when is None else f"in GW{when}"
        half = "first" if window == 1 else "second"
        unlock = (f"; the next one unlocks in GW{FIRST_HALF_LAST + 1}"
                  if window == 1 else "")
        return f"your {half}-half {name} was played {where}{unlock}"

    for other, when in played:
        if other != name and when == event:
            return (f"{other} is already recorded for GW{event} — only one chip "
                    f"may be active in a gameweek")

    if name == "freehit":
        for other, when in played:
            if other != "freehit" or when is None:
                continue
            if 0 < event - when <= FREE_HIT_MIN_GAP:
                return (f"a Free Hit was played in GW{when}, and two cannot be "
                        f"played in consecutive gameweeks")
    return None


def chip_available(chip, event: int, chip_events, first_event: int = 1) -> bool:
    return chip_blocked_reason(chip, event, chip_events, first_event) is None


def available_chips(event: int, chip_events, first_event: int = 1) -> list[str]:
    """Every chip legal to play in `event`, in CHIPS order."""
    return [c for c in CHIPS if chip_available(c, event, chip_events, first_event)]
