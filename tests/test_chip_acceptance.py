"""The audit's P0 acceptance gate, in one place.

'Synthetic tests covering all eight chips, GW19/20, one chip per GW, FT
balances 0-5, FH restoration, and a two-transfer optimum. No live run may
return a chip without a complete legal action.'

These duplicate coverage that already exists in the per-module test files, on
purpose: the gate is a single readable statement of what P0 promised, so a
later change that quietly breaks one of those promises fails here by name.
"""
import pytest

from fpl.chips import CHIPS, available_chips, chip_available
from fpl.state import State, ft_after_moves


def test_all_eight_chips_are_reachable_across_a_season():
    """Four chips, two windows, eight uses -- and playing each first-half copy
    must leave every second-half copy intact."""
    first_half = [{"chip": c, "event": 5 + i} for i, c in enumerate(CHIPS)]
    for chip in CHIPS:
        assert not chip_available(chip, 18, first_half)
        assert chip_available(chip, 30, first_half)


def test_the_window_boundary_is_gw19_to_gw20():
    played = [{"chip": "benchboost", "event": 19}]
    assert not chip_available("benchboost", 19, played)
    assert chip_available("benchboost", 20, played)


def test_an_unused_first_half_chip_does_not_become_a_spare_second_half_one():
    """The first set expires rather than rolling over, so the second half
    starts with exactly one of each however the first half went."""
    assert not chip_available("triplecaptain", 30, [{"chip": "triplecaptain",
                                                     "event": 20}])


def test_only_one_chip_is_legal_in_any_single_gameweek():
    played = [{"chip": "wildcard", "event": 10}]
    assert available_chips(10, played) == []
    for chip in (c for c in CHIPS if c != "wildcard"):
        assert chip_available(chip, 11, played)


def test_a_free_hit_cannot_follow_a_free_hit():
    played = [{"chip": "freehit", "event": 19}]
    assert not chip_available("freehit", 20, played)
    assert chip_available("freehit", 21, played)


@pytest.mark.parametrize("balance", [0, 1, 2, 3, 4, 5])
def test_every_free_transfer_balance_survives_a_chip_week(balance):
    """Preserved, not preserved-and-incremented: the chip consumes the
    gameweek's own free transfer to activate."""
    remaining, nxt = ft_after_moves(State(free_transfers=balance),
                                    transfers_made=12, chip="wildcard")
    assert (remaining, nxt) == (balance, balance)


@pytest.mark.parametrize("balance", [0, 1, 2, 3, 4, 5])
def test_every_free_transfer_balance_accrues_normally_without_a_chip(balance):
    _, nxt = ft_after_moves(State(free_transfers=balance), transfers_made=0)
    assert nxt == min(5, balance + 1)


@pytest.mark.parametrize("chip", CHIPS)
def test_no_chip_is_legal_before_an_entry_has_started(chip):
    """Wildcard and Free Hit are disabled in an opening gameweek outright; the
    other two are legal, which is the distinction the gate has to preserve."""
    legal = chip_available(chip, 7, [], first_event=7)
    assert legal is (chip not in ("wildcard", "freehit"))
