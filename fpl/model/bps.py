"""Expected bonus points.

Models BONUS ONLY. Defensive Contribution threshold points are direct
scoring and are handled in model/xp.py — the same raw actions feed both
mechanisms, but they must not be awarded twice.

R10 (audit): only the top three BPS ("bonus points system") scorers in a
MATCH earn bonus -- it is a within-match ranking, not an independent rate.
`expected_bonus`/`expected_bonus_for` below are the xP point estimate and
stay a simple fixture-scaled rate; `score_side_bps`/`award_match_bonus`
are for `model.simulate`, which draws goals/assists/clean-sheets/saves/
cards/minutes coherently per scenario and can rank them the way the real
rule does.
"""
import numpy as np
import pandas as pd

MAX_BONUS_PER_MATCH = 3.0
FIXTURE_SENSITIVITY = 0.5  # bonus is less fixture-dependent than goals

# R10 re-review (Codex) reverted a calibration that was tried and shown
# wrong here: `bonus90` is derived (model.scoring.RATE_SPECS) from the REAL
# `bonus` field FPL actually awarded each player -- i.e. it is ALREADY the
# outcome of real match-wide BPS competition, a historical per-90 average of
# points that survived it, not an abstract "how good is this player at BPS"
# rate. Projecting it forward via `bonus90 * (e_minutes/90) * scale` is an
# ordinary linear extrapolation of an already-calibrated statistic and does
# not need a further "how much survives the match" correction -- that
# survival is already baked into the historical number by construction.
# A per-position multiplier was added believing otherwise (reasoning from a
# synthetic diagnostic with an unrealistic UNIFORM bonus90 across every
# simulated player, which is not how real bonus90 is distributed across a
# real pool) and was checked against real data only after landing: applied
# to the real 2026/27 GW1-4 bootstrap, those factors cut correctly-awarded
# bonus totalling ~6.4-6.5 per match down to ~3.0 -- 46% of the true total.
# Removed entirely rather than patched, per the same principle C6-5 used
# for the earlier Beta-draw mistake: a demonstrably wrong mechanism does not
# get a smaller wrong constant, it gets removed. A genuine correction, if
# one is ever warranted, needs fitting against the production
# `blended_rates` pipeline's real historical output with an out-of-sample
# (leave-one-gameweek-out) check -- not a simulated proxy -- and is left
# as an open, undone follow-up (handoff.md R10).


def expected_bonus_for(bonus90: float, e_minutes: float, att_mult: float = 1.0) -> float:
    """Expected bonus for ONE player in ONE fixture.

    `build_xp` needs exactly this, once per (player, fixture). It was getting it
    by filtering the entire rates and minutes frames each time, which made the
    projection quadratic in squad size for no gain.
    """
    scale = 1.0 + (float(att_mult) - 1.0) * FIXTURE_SENSITIVITY
    per_match = float(bonus90) * (float(e_minutes) / 90.0) * scale
    return float(min(max(per_match, 0.0), MAX_BONUS_PER_MATCH))


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


# --- Simplified per-scenario BPS, from events the simulator already draws ---
#
# The official table scores roughly two dozen actions separately: tackles,
# clearances/blocks/interceptions, recoveries, crosses, key passes, dribbles,
# shots on/off target, passing-accuracy tiers, fouls won/conceded, offside,
# and defensive/attacking errors, on top of goals/assists/clean sheets/saves/
# cards/minutes/goals conceded. `model.simulate` only draws the latter group
# -- the rest are not modelled at all and are an open residual (handoff.md
# R10), not folded into these constants. Where an official value maps onto a
# drawn event one-to-one it is used as-is; where it does not, a documented
# blend stands in for it. Verified against the OFFICIAL 2026/27 BPS rules
# (premierleague.com, fetched 2026-09-18, after Codex's re-review flagged
# that the season had changed the save and CBI rates from what an earlier
# fetch of a season-agnostic page had shown).
BPS_APPEARANCE_SHORT = 3.0             # played, under 60 minutes (official)
BPS_APPEARANCE_LONG = 6.0              # played 60+ minutes (official)
BPS_GOAL = {"GKP": 12.0, "DEF": 12.0, "MID": 18.0, "FWD": 24.0}
# Official: non-penalty goal by position, as above; a PENALTY goal is
# always 12 regardless of position. The simulator does not distinguish a
# penalty from an open-play goal (`_allocate` shares a side's goals out by
# minutes-weighted xG90, with no penalty-taker sub-model), so this table
# always scores MID/FWD goals at the non-penalty value -- a real, and
# probably small, overestimate specifically for designated penalty takers,
# left as a residual rather than guessed at without a goal-type signal.
BPS_ASSIST = 9.0                       # official
BPS_CLEAN_SHEET = {"GKP": 12.0, "DEF": 12.0}   # official, GKP/DEF, 60+ minutes only
# 2026/27 rule: 2 BPS for ANY save, plus 1 more for a save judged a "big
# chance". The simulator has no big-chance signal for a shot faced (only
# for a chance created, and only as an official-table entry it does not
# model either), so this is the guaranteed base only -- a real, likely
# small, underestimate for busy shot-stoppers facing high-quality chances.
BPS_SAVE = 2.0
# Official splits a card -3 (yellow) / -9 (red); the simulator draws one
# undifferentiated `cards` count, so every card scores as a yellow here --
# a real, and probably small, underestimate of the rare red-card cost.
BPS_CARD = -3.0
# The official RED value, for a caller (validate_bps.py) that has the real
# yellow/red split available and does not need this approximation.
BPS_RED_CARD = -9.0
# Official, GKP/DEF only, per goal conceded (not per two, unlike the FPL
# POINTS rule in model.xp -- these are different scoring systems).
BPS_CONCEDED = -4.0
CONCEDED_BPS_POSITIONS = {"GKP", "DEF"}
# 2026/27 rules: clearance/block/interception 0.333 (one point per THREE,
# not two -- halved from the prior season), recovery 0.333 (one point per
# three, unchanged), successful tackle 2 (the SEPARATE -1-for-being-tackled
# penalty was removed this season, not the +2 for a successful one).
# `dc90` is one blended per-90 rate with no split between these three, so
# this is a documented weighted guess per position, not a precise fit.
# `scripts/validate_bps.py::logo_cv` (Codex's re-review, finding 2: fitting
# and evaluating on the SAME gameweeks is in-sample selection, not
# validation) runs leave-one-gameweek-out cross-validation over real GW1-4
# fixtures: the SAME candidate weight set was selected on every one of the
# 4 training folds (not fold-dependent, which would have signalled
# overfitting), with a mean HELD-OUT MAE of 3.24 -- close to its in-sample
# 3.26 and consistently below the uniform-0.6 baseline's 3.49 on the same
# held-out folds. Re-run that script as more gameweeks accumulate.
BPS_DC_ACTION = {"GKP": 0.5, "DEF": 0.75, "MID": 0.85, "FWD": 0.55}


def score_side_bps(positions: np.ndarray, played: np.ndarray, reached_60: np.ndarray,
                   goals: np.ndarray, assists: np.ndarray, clean_sheet: np.ndarray,
                   saves: np.ndarray, cards: np.ndarray, dc: np.ndarray,
                   conceded_on: np.ndarray) -> np.ndarray:
    """Approximate BPS per player per scenario, shape (n_players, n_sims).

    All inputs share that shape except `positions` (n_players,) and
    `clean_sheet` (n_sims,) -- broadcasting handles both. `clean_sheet` is
    whether the SIDE kept a clean sheet in that scenario; the GKP/DEF bonus
    still needs `reached_60`, same as the FPL points rule for a clean sheet.
    `conceded_on` is goals conceded while THIS player was on the pitch
    (already computed in `_score_side` for the separate FPL points penalty;
    BPS penalises every goal conceded, not every two).
    """
    is_gk_def = np.isin(positions, ("GKP", "DEF"))[:, None]
    concedes_bps = np.isin(positions, tuple(CONCEDED_BPS_POSITIONS))[:, None]
    goal_pts = np.array([BPS_GOAL.get(p, 0.0) for p in positions])[:, None]
    dc_weight = np.array([BPS_DC_ACTION.get(p, 0.6) for p in positions])[:, None]

    bps = np.where(reached_60, BPS_APPEARANCE_LONG,
                  np.where(played, BPS_APPEARANCE_SHORT, 0.0))
    bps = bps + goals * goal_pts + assists * BPS_ASSIST
    bps = bps + is_gk_def * reached_60 * clean_sheet[None, :] * 12.0
    bps = bps + saves * BPS_SAVE
    bps = bps + cards * BPS_CARD
    bps = bps + dc * dc_weight
    bps = bps + concedes_bps * conceded_on * BPS_CONCEDED
    # `rows` in `_simulate` is every player on the CLUB's roster, not just
    # the ones who appeared this gameweek -- most of a squad's fringe players
    # never take the pitch and score a genuine 0 by every component above,
    # which is indistinguishable from "actually nil" unless excluded here.
    # Left as 0 they would tie with each other for the match's TOP bps
    # (nobody else is strictly ahead of a shared 0) and `award_match_bonus`
    # would hand every one of them bonus. -inf guarantees a non-appearing
    # player is never tied for anything.
    return np.where(played, bps, -np.inf)


def award_match_bonus(bps: np.ndarray) -> np.ndarray:
    """3/2/1 bonus per player per scenario, shape matching `bps`.

    FPL's tie rule: a tied tier's SIZE, not 1, advances the next rank -- two
    players tied for first both score 3 and the next player scores 1 (rank 2
    is skipped entirely); two tied for second both score 2; two tied for
    third both score 1. A player's rank is exactly the count of players
    strictly ahead of him plus one, so that count alone -- 0, 1 or 2 -- fixes
    his award regardless of how the tiers above him are sized or tied.
    """
    strictly_ahead = (bps[None, :, :] > bps[:, None, :]).sum(axis=1)
    award = np.select([strictly_ahead == 0, strictly_ahead == 1, strictly_ahead == 2],
                      [3.0, 2.0, 1.0], default=0.0)
    # A non-appearing player (see `score_side_bps`) is marked -inf so he can
    # never OUTRANK a real scorer -- but if fewer than three players in the
    # match have a real score, several -inf players still TIE with each
    # other and, on `strictly_ahead` alone, would inherit whatever rank the
    # real scorers left open. No one who did not play may ever receive
    # bonus, regardless of how few genuine candidates the match has.
    return np.where(np.isneginf(bps), 0.0, award)
