"""Rank utility -- the objective that matches "win the league".

Maximising expected points is the correct objective for a game scored on
points, and FPL is not one: it is scored on RANK against several million other
managers. The two objectives come apart in one specific, decisive way. Points
you score that the field also scores move you nowhere. A 60%-owned striker's
hat-trick is worth three goals to you and to most of the people you are trying
to beat, so it changes your score a lot and your rank almost not at all.

An expected-points model cannot see that, because the difference lives
entirely in the JOINT distribution of your score and the field's. So the field
is drawn here from the SAME simulated matches as your own squad
(`model.simulate`): when you and a rival both own the striker, his haul lands
on both sides of the comparison and cancels, exactly as it does in reality.

Building this refuted a piece of folk wisdom worth stating plainly, because
the codebase acted on it before the simulation could check it. "Captain the
ceiling, not the mean" is FALSE for expected rank. Between a player who
returns exactly 6 and one who returns 0 or 20 at the same mean of 6.0, the
steady one wins on mean rank percentile in every standing tested -- chasing,
level and defending alike -- because a 70% blank costs more rank than a 30%
haul buys. Variance does not raise an average.

Where variance genuinely pays is a THRESHOLD objective: `p_beat_target`. If
the score you need is beyond what your steady option can reach, the volatile
one is the only route to it, and the crossover is sharp -- with a tight squad
around 33 points and a 6-point armband, the steady captain wins every bar up
to 40 and loses every bar from 42 up. That is the real shape of the trade:
take variance when, and only when, you cannot reach the target without it.
Which is why `target` is a parameter here and not a hard-coded preference.

The effect the simulation captures unambiguously is CORRELATION. Three
defenders from one club are one clean-sheet bet placed three times, and an
expectation-only model prices them as three independent bets no matter what.
"""
import numpy as np
import pandas as pd

# Rival managers drawn per evaluation. A rank percentile is a mean over these,
# so the standard error goes as 1/sqrt(RIVALS); 400 puts it near 2.5%, which
# is finer than the differences worth acting on.
RIVALS = 400
XI_SIZE = 11
CAPTAIN_MULTIPLIER = 2
# Legal FPL formations as (DEF, MID, FWD). One goalkeeper always.
FORMATIONS = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 5, 1), (4, 3, 3),
              (5, 4, 1), (5, 3, 2), (5, 2, 3)]


def field_weights(xp_df: pd.DataFrame) -> np.ndarray:
    """How many times the AVERAGE manager counts each player's points.

    Sums to 12: eleven starters plus the armband. This is effective ownership
    in the FPL sense -- the share of the field collecting a player's score --
    and it is what your own score has to be measured against rather than
    against zero.
    """
    own = xp_df["ownership"].astype(float).clip(lower=0.0).to_numpy()
    playing = (xp_df["p_play"].astype(float).to_numpy()
               if "p_play" in xp_df.columns else np.ones(len(xp_df)))
    raw = own * playing
    total = raw.sum()
    if total <= 0:
        return np.zeros(len(xp_df))
    starters = raw / total * XI_SIZE
    return starters + _captain_shares(xp_df)


def _captain_shares(xp_df: pd.DataFrame) -> np.ndarray:
    """How the field's armband is distributed. Sums to 1.

    Captaincy is far more concentrated than ownership -- the field piles onto
    two or three names -- so this is ownership weighted by projection rather
    than ownership alone.
    """
    own = xp_df["ownership"].astype(float).clip(lower=0.0).to_numpy()
    xp = xp_df["xp_next1"].astype(float).clip(lower=0.0).to_numpy()
    pull = own * xp ** 3          # cubed: the armband concentrates hard
    total = pull.sum()
    return pull / total if total > 0 else np.zeros(len(xp_df))


def _inclusion_probabilities(weights: np.ndarray, k: int,
                             ceiling: np.ndarray | None = None) -> np.ndarray:
    """Marginal P(selected) for each unit, scaled to select exactly `k`.

    Units whose share would exceed their `ceiling` are pinned there and the
    remaining mass is redistributed. That is what stops a 71%-owned striker
    from being diluted by the long tail of forwards nobody picks -- and,
    through the ceiling, what stops him being started by more managers than
    own him, which is not a thing that can happen.
    """
    w = np.asarray(weights, dtype=float).clip(min=0.0)
    k = int(min(k, len(w)))
    if k <= 0 or w.sum() <= 0:
        return np.zeros(len(w))
    cap = (np.ones(len(w)) if ceiling is None
           else np.clip(np.asarray(ceiling, dtype=float), 0.0, 1.0))
    pi = np.minimum(w / w.sum() * k, cap)
    for _ in range(len(w)):
        over = pi >= cap
        if not over.any() or over.all():
            break
        spare = k - cap[over].sum()
        rest = ~over
        rest_total = w[rest].sum()
        if rest_total <= 0 or spare <= 0:
            break
        scaled = pi.copy()
        scaled[over] = cap[over]
        scaled[rest] = np.minimum(w[rest] / rest_total * spare, cap[rest])
        if np.allclose(scaled, pi):
            break
        pi = scaled
    return np.clip(pi, 0.0, cap)


def _systematic_pps(pi: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exactly `k` indices whose marginal selection rates equal `pi`.

    Systematic probability-proportional-to-size sampling: shuffle, walk the
    cumulative inclusion probabilities and take a unit every 1.0 of mass from a
    random start. Weighted sampling WITHOUT replacement does not have this
    property -- its marginals are pulled toward uniform by the size of the
    pool, which made the simulated field far weaker than the real one.

    The marks are spaced by `pi.sum() / k` rather than by 1.0. Those agree when
    the probabilities already sum to `k`, and when a ceiling has held the total
    below `k` the spacing keeps the walk covering the WHOLE range: fixed unit
    steps leave the fractional tail beyond the last mark unreachable, which
    silently inflates everyone before it.
    """
    total = float(pi.sum())
    if k <= 0 or total <= 0:
        return np.array([], dtype=int)
    step = total / k
    order = rng.permutation(len(pi))
    cum = np.cumsum(pi[order])
    marks = rng.random() * step + np.arange(k) * step
    return order[np.clip(np.searchsorted(cum, marks), 0, len(pi) - 1)]


def sample_rival_squads(xp_df: pd.DataFrame, n_rivals: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw `n_rivals` plausible starting XIs, shape (n_rivals, n_players).

    Entries are 0 (not started), 1 (started) or 2 (captained). Players are
    drawn by systematic PPS into a randomly chosen legal formation, so each
    one's rate of appearing MATCHES HIS OWNERSHIP -- the field looks like the
    field: heavy on the template, occasionally carrying a differential.

    Sampling whole rival squads rather than scoring against one averaged
    "field team" matters: an average of many managers has far less variance
    than any actual manager, which would make every squad look closer to the
    middle than it is.
    """
    own = xp_df["ownership"].astype(float).clip(lower=1e-6).to_numpy()
    xp = xp_df["xp_next1"].astype(float).to_numpy()
    pos = xp_df["position"].to_numpy()
    # Owning a player and STARTING him are different things: a manager owns 15
    # and plays 11, benching the cheap fodder. Weighting slots by ownership
    # alone spreads them evenly over everyone owned, which left a 71%-owned
    # striker in a third of rival XIs and made the simulated field score ~44
    # against a real FPL average nearer 50-55. Ownership times projection
    # concentrates the slots on the players managers actually field.
    start_pull = own * np.clip(xp, 0.0, None)
    by_pos = {p: np.flatnonzero(pos == p) for p in ("GKP", "DEF", "MID", "FWD")}

    out = np.zeros((n_rivals, len(xp_df)))
    for r in range(n_rivals):
        n_def, n_mid, n_fwd = FORMATIONS[rng.integers(len(FORMATIONS))]
        picked = []
        for position, k in (("GKP", 1), ("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
            pool = by_pos[position]
            k = min(k, len(pool))
            if k == 0:
                continue
            # A manager can only start a player he owns.
            pi = _inclusion_probabilities(start_pull[pool], k,
                                          ceiling=own[pool] / 100.0)
            chosen = np.unique(_systematic_pps(pi, k, rng))
            # The ownership ceiling can leave a thin position short of its
            # slots; a real manager still fields someone, so the remainder is
            # topped up from whoever else is available.
            if len(chosen) < k:
                spare = np.setdiff1d(np.arange(len(pool)), chosen)
                # Anyone already pinned at his ownership ceiling is appearing at
                # exactly the right rate; topping him up again would start him
                # for managers who never bought him.
                pinned = pi >= (own[pool] / 100.0) - 1e-9
                free = spare[~pinned[spare]]
                spare = free if len(free) else spare
                if len(spare):
                    w = start_pull[pool][spare]
                    w = w / w.sum() if w.sum() > 0 else None
                    extra = rng.choice(spare, size=min(k - len(chosen), len(spare)),
                                       replace=False, p=w)
                    chosen = np.concatenate([chosen, np.atleast_1d(extra)])
            picked.extend(pool[chosen])
        picked = np.array(picked, dtype=int)
        out[r, picked] = 1.0
        # Managers captain the best player they own, which is what makes the
        # field's armband concentrate on a handful of names.
        out[r, picked[int(np.argmax(xp[picked]))]] = CAPTAIN_MULTIPLIER
    return out


def squad_scores(squads: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Gameweek scores, shape (n_squads, n_sims).

    `squads` carries the 0/1/2 multipliers, so this is a plain matrix product
    against the simulated player points -- and because both sides come from
    ONE set of draws, shared ownership cancels where it should.
    """
    return np.asarray(squads, dtype=float) @ np.asarray(samples, dtype=float)


def rank_percentile(my_scores: np.ndarray, rival_scores: np.ndarray) -> float:
    """Share of rival manager-gameweeks this squad finishes ahead of.

    0.5 is dead average; 0.9 means beating nine rivals in ten. Ties split,
    which matters because a squad identical to a rival's ties rather than wins.
    """
    mine = np.asarray(my_scores, dtype=float)[None, :]
    rivals = np.asarray(rival_scores, dtype=float)
    return float((np.mean(mine > rivals) + 0.5 * np.mean(mine == rivals)))


def p_beat_target(my_scores: np.ndarray, rival_scores: np.ndarray,
                  target: float = 0.5) -> float:
    """P(this squad beats the `target` quantile of the field in a gameweek).

    `target=0.5` is "beat the median manager" and tracks expected points
    closely. `target=0.9` is "have the kind of week that puts you in the top
    tenth", which is the objective a green arrow actually rewards -- and the
    only regime in which taking variance is correct.

    The bar is recomputed per simulation rather than once, so it moves with
    the week: a gameweek in which the template hauls raises what it takes to
    beat the field, which is precisely when a differential has to fire.
    """
    bar = np.quantile(np.asarray(rival_scores, dtype=float), float(target), axis=0)
    return float(np.mean(np.asarray(my_scores, dtype=float) > bar))


def best_captain_by_rank(xi: list[int], ids: list[int], samples: np.ndarray,
                         rival_scores: np.ndarray, target: float = 0.5) -> int:
    """The armband maximising P(beat the `target` quantile of the field).

    Evaluated exhaustively -- eleven candidates is nothing. At the default
    target this lands on essentially the highest expected points, which is the
    honest answer; raise `target` and it will start paying for a ceiling.
    """
    row = {pid: i for i, pid in enumerate(ids)}
    base = sum(samples[row[p]] for p in xi if p in row)
    best, best_p = None, -1.0
    for pid in xi:
        if pid not in row:
            continue
        p = p_beat_target(base + samples[row[pid]], rival_scores, target)
        if p > best_p:
            best, best_p = pid, p
    return best


def squad_indicator(starting_ids, captain, ids) -> np.ndarray:
    """0/1/2 multipliers for one XI, in `ids` order."""
    row = {pid: i for i, pid in enumerate(ids)}
    out = np.zeros(len(ids))
    for pid in starting_ids:
        if pid in row:
            out[row[pid]] = 1.0
    if captain in row:
        out[row[captain]] = float(CAPTAIN_MULTIPLIER)
    return out


def score_candidate(squad, ids: list[int], samples: np.ndarray,
                    rival_scores: np.ndarray, target: float = 0.5) -> dict:
    """How one candidate squad actually fares against the simulated field.

    The armband is re-chosen per candidate, because the best captain in a
    squad is a property of that squad and not of the pool.
    """
    captain = best_captain_by_rank(list(squad.starting_ids), ids, samples,
                                   rival_scores, target)
    mine = squad_scores(squad_indicator(squad.starting_ids, captain, ids)[None, :],
                        samples)[0]
    return {
        "captain": captain,
        "p_beat_target": p_beat_target(mine, rival_scores, target),
        "rank_percentile": rank_percentile(mine, rival_scores),
        "mean_points": float(mine.mean()),
        "sd_points": float(mine.std()),
    }


def pick_best_squad(candidates: list, ids: list[int], samples: np.ndarray,
                    rival_scores: np.ndarray, target: float = 0.5):
    """(best squad, [scores for every candidate]) by P(beating the field).

    Ties break on expected points, so when the simulation cannot separate two
    squads the answer falls back to the projection rather than to sampling
    noise -- which matters, because a rank percentile over a few hundred
    rivals carries a standard error of its own.
    """
    if not candidates:
        raise ValueError(
            "no candidate squads to rank -- the solver returned nothing to "
            "choose between"
        )
    scored = [score_candidate(c, ids, samples, rival_scores, target)
              for c in candidates]
    best = max(range(len(candidates)),
               key=lambda i: (scored[i]["p_beat_target"], scored[i]["mean_points"]))
    return candidates[best], scored
