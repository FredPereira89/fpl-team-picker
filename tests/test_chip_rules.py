from fpl.chips import (CHIPS, FIRST_HALF_LAST, chip_window, chip_available,
                       chip_blocked_reason, available_chips)


def _used(chip, event):
    return {"chip": chip, "event": event}


def test_the_season_splits_into_two_chip_windows():
    assert chip_window(1) == 1
    assert chip_window(FIRST_HALF_LAST) == 1
    assert chip_window(FIRST_HALF_LAST + 1) == 2
    assert chip_window(38) == 2


def test_a_first_half_wildcard_still_leaves_the_second_half_one():
    """Two of every chip a season: one for GW1-19, one for GW20-38. A name-only
    record could not tell them apart and suppressed the second for good."""
    played = [_used("wildcard", 5)]
    assert not chip_available("wildcard", 8, played)
    assert chip_available("wildcard", 25, played)


def test_an_unused_first_half_chip_does_not_roll_over():
    """The first set EXPIRES at GW19. Playing the second-half Wildcard in GW20
    must not leave a spare one from a first half that was never used."""
    played = [_used("wildcard", 20)]
    assert not chip_available("wildcard", 25, played)


def test_wildcard_and_free_hit_are_blocked_in_the_opening_gameweek():
    """Transfers before the opening deadline are already unlimited, so FPL
    disables both chips there."""
    assert not chip_available("wildcard", 1, [])
    assert not chip_available("freehit", 1, [])
    assert chip_available("benchboost", 1, [])
    assert chip_available("triplecaptain", 1, [])
    assert chip_available("wildcard", 2, [])


def test_an_entry_that_joined_late_has_its_own_opening_gameweek():
    assert not chip_available("wildcard", 7, [], first_event=7)
    assert chip_available("wildcard", 8, [], first_event=7)


def test_free_hits_cannot_be_played_in_consecutive_gameweeks():
    played = [_used("freehit", 19)]
    assert not chip_available("freehit", 20, played)
    assert chip_available("freehit", 21, played)


def test_only_one_chip_may_be_active_in_a_gameweek():
    played = [_used("benchboost", 12)]
    assert not chip_available("triplecaptain", 12, played)
    assert chip_available("triplecaptain", 13, played)


def test_an_undated_record_is_charged_to_the_first_half():
    """State files written before chip_events existed know WHICH chip was
    played but not when. Charging it to the first half can only forfeit a chip
    already spent; charging it to the second would forfeit one still held."""
    played = [{"chip": "benchboost", "event": None}]
    assert not chip_available("benchboost", 10, played)
    assert chip_available("benchboost", 25, played)


def test_api_chip_names_are_canonicalised():
    played = [_used("3xc", 4)]
    assert not chip_available("triplecaptain", 6, played)


def test_available_chips_lists_everything_legal_this_week():
    assert available_chips(2, []) == list(CHIPS)
    assert available_chips(1, []) == ["benchboost", "triplecaptain"]


def test_a_blocked_chip_explains_itself():
    reason = chip_blocked_reason("wildcard", 8, [_used("wildcard", 5)])
    assert reason and "GW5" in reason
    assert chip_blocked_reason("wildcard", 25, [_used("wildcard", 5)]) is None
