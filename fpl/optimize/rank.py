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
from itertools import combinations, product

import numpy as np
import pandas as pd

from .squad import XI_SIZE, XI_MIN, XI_MAX, MAX_PER_CLUB

# Rival managers drawn per evaluation. A rank percentile is a mean over these,
# so the standard error goes as 1/sqrt(RIVALS); 400 puts it near 2.5%, which
# is finer than the differences worth acting on.
RIVALS = 400
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
    if not (total > 0):
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
    if k <= 0 or not (w.sum() > 0):
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
        if not (rest_total > 0) or not (spare > 0):
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
    if k <= 0 or not (total > 0):
        return np.array([], dtype=int)
    step = total / k
    order = rng.permutation(len(pi))
    cum = np.cumsum(pi[order])
    marks = rng.random() * step + np.arange(k) * step
    return order[np.clip(np.searchsorted(cum, marks), 0, len(pi) - 1)]


# What a rival XI may cost: budget minus the cheapest legal bench that
# completes it, checked exactly per-XI by `_cheapest_legal_bench` below.
# An earlier version bounded this with a single pool-wide scalar (cheapest
# keeper + 3 cheapest outfielders in the whole pool, regardless of formation,
# club cap, or whether those cheap players were already in the XI) -- that
# passed XIs whose true cheapest bench blew the budget once the actual
# formation and club cap were accounted for.
DEFAULT_BUDGET = 100.0
REPAIR_PASSES = 20


def _cheapest_legal_bench(xi, pos, club, price, by_pos_sorted) -> float | None:
    """The cheapest legal bench (position-exact, club-cap-respecting,
    distinct from the XI) that completes `xi` into a real fifteen, or None
    if this pool cannot supply one.

    A 3-5-2 XI needs 2 DEF + 0 MID + 1 FWD on the bench, not "any 3
    outfielders" -- and any cheap player already IN the XI can't also fill
    a bench slot. The club cap is the reason this cannot be filled
    position-by-position, cheapest-first, and stop there: the cap is shared
    ACROSS positions, so taking the cheapest reserve keeper from a club
    already near the cap can use up the only room a later position's one
    remaining candidate needed. An earlier version did exactly that --
    greedy per position, no backtracking -- and could both wrongly report a
    bench as impossible (there was a pricier keeper that left room) and,
    when it did find one, not find the cheapest one, since a cheaper
    combination one slot back was never reconsidered.

    This runs an exact search instead: every still-needed bench place across
    every position becomes one SLOT (a 3-5-2 XI needing 2 more DEF is two
    DEF slots, not one), and depth-first search tries each slot's candidates
    cheapest-first, backtracking when a club-cap conflict blocks a later
    slot. There are at most 4 bench slots in total, so this stays fast in
    practice even though it is an exact search rather than a linear scan --
    PROVIDED the bound that prunes a branch is tight. Comparing only the
    cost spent SO FAR against the best full solution found is not tight
    enough: real FPL prices repeat heavily (whole tiers of players share a
    price point), so many candidates tie on cost, and a partial path's own
    cost does not reach the full-solution bound until the LAST slot is
    filled -- meaning every one of those tied branches gets walked to full
    depth before the tie finally prunes it, which measured over a second
    per call on a pool with several dozen same-priced candidates per
    position. The bound instead adds the cheapest possible cost of every
    SLOT STILL TO FILL (each slot's own cheapest candidate, ignoring club
    conflicts -- an admissible underestimate, since the true cost can only
    be that or higher) to the cost already spent, so a branch that can
    provably not beat the incumbent is cut at whatever depth it stops being
    competitive, not only at the end.
    `by_pos_sorted[position]` must already be sorted by price ascending.
    """
    from .squad import SQUAD_SPLIT
    xi_set = {int(i) for i in xi}
    base_club_count: dict = {}
    for i in xi_set:
        base_club_count[club[i]] = base_club_count.get(club[i], 0) + 1

    slots = []
    for position, need_total in SQUAD_SPLIT.items():
        in_xi = sum(1 for i in xi_set if pos[i] == position)
        need = need_total - in_xi
        if need <= 0:
            continue
        candidates = [i for i in by_pos_sorted.get(position, ()) if i not in xi_set]
        slots.extend([candidates] * need)
    if not slots:
        return 0.0

    # suffix_min[k] = cheapest possible sum of slots[k:], each slot's own
    # cheapest candidate regardless of club -- an admissible lower bound on
    # what filling the rest could cost from any state at that depth.
    suffix_min = [0.0] * (len(slots) + 1)
    for k in range(len(slots) - 1, -1, -1):
        cheapest = price[slots[k][0]] if slots[k] else 0.0
        suffix_min[k] = suffix_min[k + 1] + cheapest

    best = [None]

    def search(slot_idx: int, club_count: dict, cost: float, used: set) -> None:
        if best[0] is not None and cost + suffix_min[slot_idx] >= best[0]:
            return                                  # this branch cannot improve on best
        if slot_idx == len(slots):
            best[0] = cost
            return
        for i in slots[slot_idx]:
            if i in used:
                continue
            c = club[i]
            if club_count.get(c, 0) >= MAX_PER_CLUB:
                continue
            used.add(i)
            club_count[c] = club_count.get(c, 0) + 1
            search(slot_idx + 1, club_count, cost + price[i], used)
            club_count[c] -= 1
            used.discard(i)

    search(0, dict(base_club_count), 0.0, set())
    return best[0]


def _repair(picked: np.ndarray, pos, club, price, start_pull, by_pos,
           by_pos_sorted, budget: float, rng: np.random.Generator) -> np.ndarray:
    """Swap players out of a drawn XI until it is a squad someone could own.

    Two rules a starting eleven inherits from the fifteen it came from: no
    more than MAX_PER_CLUB from one club, and a price that leaves room for a
    LEGAL bench -- checked exactly, via `_cheapest_legal_bench`, not the
    pool-wide approximation. Each pass replaces one offender -- the
    cheapest-pull player of an over-represented club, else the dearest
    player -- with a same-position player drawn by the same start pull from
    those who do not re-offend.

    That fast heuristic is deterministic about WHICH player it removes, and
    narrows who it can replace him with -- cheap, and almost always enough,
    but it can cycle: two players with no other legal partner swap for each
    other forever, since removing the dearest player always names the SAME
    player and his only legal replacement is always the SAME other player.
    No number of further passes escapes a deterministic 2-cycle on its own.
    Every visited state is tracked; the moment the heuristic's proposed move
    would repeat one, the player removed is instead drawn UNIFORMLY AT
    RANDOM from the whole XI rather than always the same offender -- cheap
    (no extra legality search), and enough randomness that repeating the
    exact same cycle again is very unlikely. If REPAIR_PASSES still runs out,
    the loop gives up and returns the best it reached, which keeps the
    sampler total rather than perfect on a pathological pool.
    """
    picked = list(int(i) for i in picked)

    def propose(out, too_dear):
        remaining = [i for i in picked if i != out]
        # Excludes `out` itself, not only `remaining` -- `out` is not IN
        # `remaining` by construction, so leaving it in `pool` let the
        # "swap" re-pick the very player being removed: a no-op disguised
        # as progress, since removing `out` from its own club's count
        # trivially makes room for `out` again.
        pool = [i for i in by_pos[pos[out]] if i not in remaining and i != out]
        counts_now: dict = {}
        for i in remaining:
            counts_now[club[i]] = counts_now.get(club[i], 0) + 1
        # Club first; among club-legal candidates prefer the cheaper half
        # when cost was the problem. This is a HEURISTIC search step, not the
        # proof -- the stopping condition above re-checks exactly on the next
        # pass, so an imperfect pick here just costs another iteration.
        club_ok = [i for i in pool if counts_now.get(club[i], 0) < MAX_PER_CLUB]
        if not club_ok:
            club_ok = pool
        if too_dear and price is not None and len(club_ok) > 1:
            club_ok = sorted(club_ok, key=lambda i: price[i])[:max(1, len(club_ok) // 2)]
        if not club_ok:
            return None
        w = np.array([max(start_pull[i], 1e-12) for i in club_ok])
        new = int(rng.choice(club_ok, p=w / w.sum()))
        return remaining + [new]

    seen: set = set()
    for _ in range(REPAIR_PASSES):
        counts: dict = {}
        for i in picked:
            counts[club[i]] = counts.get(club[i], 0) + 1
        over = [c for c, n in counts.items() if n > MAX_PER_CLUB]
        too_dear = False
        if price is not None:
            bench_cost = _cheapest_legal_bench(picked, pos, club, price, by_pos_sorted)
            too_dear = bench_cost is None or sum(price[i] for i in picked) + bench_cost > budget
        if not over and not too_dear:
            break
        seen.add(frozenset(picked))
        if over:
            candidates = [i for i in picked if club[i] == over[0]]
            out = min(candidates, key=lambda i: start_pull[i])
        else:
            out = max(picked, key=lambda i: price[i])
        candidate = propose(out, too_dear)
        if candidate is None or frozenset(candidate) in seen:
            out = int(rng.choice(picked))
            candidate = propose(out, too_dear)
        if candidate is not None:
            picked = candidate
    return np.array(picked, dtype=int)


def sample_rival_squads(xp_df: pd.DataFrame, n_rivals: int,
                        rng: np.random.Generator,
                        budget: float = DEFAULT_BUDGET) -> np.ndarray:
    """Draw `n_rivals` plausible starting XIs, shape (n_rivals, n_players).

    Entries are 0 (not started), 1 (started) or 2 (captained). Players are
    drawn by systematic PPS into a randomly chosen legal formation, so each
    one's rate of appearing MATCHES HIS OWNERSHIP -- the field looks like the
    field: heavy on the template, occasionally carrying a differential.

    Every XI is then made LEGAL: at most MAX_PER_CLUB from one club, and a
    price that leaves room for a bench under `budget`. Ownership-weighted
    draws alone produced elevens of premiums no manager could own together,
    which overstated the field's strength and its correlation.

    Sampling whole rival squads rather than scoring against one averaged
    "field team" matters: an average of many managers has far less variance
    than any actual manager, which would make every squad look closer to the
    middle than it is.
    """
    own = xp_df["ownership"].astype(float).clip(lower=1e-6).to_numpy()
    xp = xp_df["xp_next1"].astype(float).to_numpy()
    pos = xp_df["position"].to_numpy()
    club = (xp_df["team"].to_numpy() if "team" in xp_df.columns
            else np.arange(len(xp_df)))          # no club column: nothing to cap
    price = (xp_df["price"].astype(float).to_numpy() if "price" in xp_df.columns
             else None)
    # Owning a player and STARTING him are different things: a manager owns 15
    # and plays 11, benching the cheap fodder. Weighting slots by ownership
    # alone spreads them evenly over everyone owned, which left a 71%-owned
    # striker in a third of rival XIs and made the simulated field score ~44
    # against a real FPL average nearer 50-55. Ownership times projection
    # concentrates the slots on the players managers actually field.
    start_pull = own * np.clip(xp, 0.0, None)
    by_pos = {p: np.flatnonzero(pos == p) for p in ("GKP", "DEF", "MID", "FWD")}
    # Sorted once here, not per repair pass: `_cheapest_legal_bench` runs up to
    # REPAIR_PASSES times per rival, so a fresh sort each call would turn an
    # O(n log n) cost into the dominant cost of the whole sampler.
    by_pos_sorted = ({p: idx[np.argsort(price[idx])] for p, idx in by_pos.items()}
                     if price is not None else {})

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
        picked = _repair(np.array(picked, dtype=int), pos, club, price, start_pull,
                         by_pos, by_pos_sorted, budget, rng)
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
    return _clear_rate(np.asarray(my_scores, dtype=float), bar)


def _clear_rate(mine: np.ndarray, bar: np.ndarray) -> float:
    """P(clear the bar), with a tie worth half.

    A strict `>` gave no credit for landing exactly on the bar, which in whole
    FPL points happens often -- and most often to template-heavy squads, whose
    scores cluster with the field's. Tied classic-league teams share a
    position after the transfer tiebreak, so half credit is the honest value.
    """
    return float(np.mean(mine > bar) + 0.5 * np.mean(mine == bar))


def best_captain_by_rank(xi: list[int], ids: list[int], samples: np.ndarray,
                         rival_scores: np.ndarray, target: float = 0.5,
                         bar: np.ndarray | None = None) -> int:
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
        total = base + samples[row[pid]]
        p = (p_beat_bar(total, bar) if bar is not None
             else p_beat_target(total, rival_scores, target))
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


def _lineup_base_scores(lineup, ids: list[int], samples: np.ndarray,
                        played: np.ndarray, positions) -> np.ndarray:
    """Scenario scores before captaincy, including legal automatic subs.

    `played` is deliberately separate from `samples`: zero points does not
    mean zero minutes.  An appearing starter who scores zero stays in the XI;
    only a player with `played == False` can be replaced.

    The reserve goalkeeper can replace only the starting goalkeeper.  The
    three outfield substitutes are considered in bench order.  For each
    scenario we choose the largest legal set of replacements, then the
    lexicographically earliest set by bench priority.  There are at most three
    outfield bench players, so exhaustively checking their eight subsets is
    both exact and cheap.
    """
    samples = np.asarray(samples, dtype=float)
    played = np.asarray(played, dtype=bool)
    if samples.shape != played.shape:
        raise ValueError("played mask must have the same shape as samples")

    row = {int(pid): i for i, pid in enumerate(ids)}
    if hasattr(positions, "get"):
        position_of = lambda pid: str(positions.get(int(pid)))
    else:
        position_of = lambda pid: str(positions[row[int(pid)]])

    xi = [int(pid) for pid in lineup.xi]
    bench = [int(pid) for pid in lineup.bench]
    missing = [pid for pid in xi + bench if pid not in row]
    if missing:
        raise ValueError(f"lineup players missing from simulation: {missing}")

    xi_rows = [row[pid] for pid in xi]
    base = np.where(played[xi_rows], samples[xi_rows], 0.0).sum(axis=0)

    # Goalkeeper substitution is independent of the outfield formation.
    starting_gk = next((pid for pid in xi if position_of(pid) == "GKP"), None)
    bench_gk = next((pid for pid in bench if position_of(pid) == "GKP"), None)
    if starting_gk is not None and bench_gk is not None:
        use_gk = ~played[row[starting_gk]] & played[row[bench_gk]]
        base = base + np.where(use_gk, samples[row[bench_gk]], 0.0)

    outfield_positions = ("DEF", "MID", "FWD")
    starting_outfield = [pid for pid in xi if position_of(pid) != "GKP"]
    bench_outfield = [pid for pid in bench if position_of(pid) != "GKP"]
    if not bench_outfield:
        return base

    start_counts = {
        pos: sum(position_of(pid) == pos for pid in starting_outfield)
        for pos in outfield_positions
    }
    missing_counts = {
        pos: sum((~played[row[pid]] for pid in starting_outfield
                  if position_of(pid) == pos), np.zeros(samples.shape[1], dtype=int))
        for pos in outfield_positions
    }

    # Prefer fielding as many players as possible; among equally full legal
    # outcomes, honour the manager's bench order.
    subsets = []
    for size in range(len(bench_outfield) + 1):
        subsets.extend(combinations(range(len(bench_outfield)), size))
    subsets.sort(key=lambda subset: (
        -len(subset),
        tuple(-int(i in subset) for i in range(len(bench_outfield))),
    ))

    assigned = np.zeros(samples.shape[1], dtype=bool)
    for subset in subsets:
        if not subset:
            continue
        selected = [bench_outfield[i] for i in subset]
        selected_counts = {
            pos: sum(position_of(pid) == pos for pid in selected)
            for pos in outfield_positions
        }

        # Which positions can the selected bench players replace while leaving
        # the nominal XI in a legal FPL formation?  Unreplaced DNP starters
        # remain empty slots; they do not license an otherwise-illegal shape.
        removal_patterns = []
        for removals in product(range(len(subset) + 1), repeat=3):
            if sum(removals) != len(subset):
                continue
            final = {
                pos: start_counts[pos] - removals[j] + selected_counts[pos]
                for j, pos in enumerate(outfield_positions)
            }
            if all(XI_MIN[pos] <= final[pos] <= XI_MAX[pos]
                   for pos in outfield_positions):
                removal_patterns.append(removals)
        if not removal_patterns:
            continue

        available = np.ones(samples.shape[1], dtype=bool)
        for pid in selected:
            available &= played[row[pid]]
        replaceable = np.zeros(samples.shape[1], dtype=bool)
        for removals in removal_patterns:
            possible = np.ones(samples.shape[1], dtype=bool)
            for j, pos in enumerate(outfield_positions):
                possible &= missing_counts[pos] >= removals[j]
            replaceable |= possible

        use = available & replaceable & ~assigned
        if np.any(use):
            sub_points = samples[[row[pid] for pid in selected]].sum(axis=0)
            base = base + np.where(use, sub_points, 0.0)
            assigned |= use
        if np.all(assigned):
            break
    return base


def lineup_scores(lineup, ids: list[int], samples: np.ndarray,
                  played: np.ndarray, positions, captain: int | None = None,
                  vice: int | None = None) -> np.ndarray:
    """Score the reported lineup under FPL autosub and armband rules."""
    samples = np.asarray(samples, dtype=float)
    played = np.asarray(played, dtype=bool)
    captain = int(lineup.captain if captain is None else captain)
    vice = int(lineup.vice if vice is None else vice)
    xi = {int(pid) for pid in lineup.xi}
    if captain not in xi or vice not in xi or captain == vice:
        raise ValueError("captain and vice must be distinct members of the starting XI")

    row = {int(pid): i for i, pid in enumerate(ids)}
    base = _lineup_base_scores(lineup, ids, samples, played, positions)
    cap_row, vice_row = row[captain], row[vice]
    bonus = np.where(played[cap_row], samples[cap_row],
                     np.where(played[vice_row], samples[vice_row], 0.0))
    return base + bonus


def best_armband_by_rank(lineup, ids: list[int], samples: np.ndarray,
                         played: np.ndarray, positions,
                         rival_scores: np.ndarray, target: float = 0.5,
                         bar: np.ndarray | None = None) -> tuple[int, int]:
    """Exhaustively choose the captain/vice pair for the rank objective."""
    samples = np.asarray(samples, dtype=float)
    played = np.asarray(played, dtype=bool)
    base = _lineup_base_scores(lineup, ids, samples, played, positions)
    row = {int(pid): i for i, pid in enumerate(ids)}
    best = None
    for captain in lineup.xi:
        for vice in lineup.xi:
            if int(vice) == int(captain):
                continue
            cap_row, vice_row = row[int(captain)], row[int(vice)]
            bonus = np.where(played[cap_row], samples[cap_row],
                             np.where(played[vice_row], samples[vice_row], 0.0))
            total = base + bonus
            p = (p_beat_bar(total, bar) if bar is not None
                 else p_beat_target(total, rival_scores, target))
            key = (p, float(total.mean()), -int(captain), -int(vice))
            if best is None or key > best[0]:
                best = (key, int(captain), int(vice))
    if best is None:
        raise ValueError("no legal captain/vice pair in lineup")
    return best[1], best[2]


def score_candidate(squad, ids: list[int], samples: np.ndarray,
                    rival_scores: np.ndarray, target: float = 0.5,
                    bar: np.ndarray | None = None, penalty: float = 0.0,
                    captain: int | None = None, vice: int | None = None,
                    lineup=None, played: np.ndarray | None = None,
                    positions=None) -> dict:
    """How one candidate squad actually fares against the simulated field.

    The armband is re-chosen per candidate by default, because the best
    captain in a squad is a property of that squad and not of the pool. Pass
    `captain` to score a SPECIFIC armband instead -- the caller already knows
    it (e.g. the production captain a lineup builder chose on its own terms)
    and wants these statistics to describe that captain rather than rank's
    own preferred one. Reporting rank's captain's numbers next to a different
    reported captain was C6-2's remaining gap: the XI matched, the armband
    did not, so mean_points/p_beat_target/rank_percentile described a week
    that was never actually fielded.

    `penalty` is subtracted from every simulated week. For a transfer plan it
    is the points hit: without it a plan costing -4 would be compared against
    the field on the same terms as one costing nothing, which quietly makes
    hits free.
    """
    if lineup is not None:
        if played is None or positions is None:
            raise ValueError("lineup scoring requires played masks and positions")
        if captain is None:
            captain, vice = best_armband_by_rank(
                lineup, ids, samples, played, positions, rival_scores,
                target=target, bar=bar)
        elif vice is None:
            vice = int(lineup.vice)
        mine = lineup_scores(lineup, ids, samples, played, positions,
                             captain=int(captain), vice=int(vice)) - float(penalty)
    else:
        if captain is None:
            captain = best_captain_by_rank(list(squad.starting_ids), ids, samples,
                                           rival_scores, target, bar=bar)
        mine = squad_scores(squad_indicator(squad.starting_ids, captain, ids)[None, :],
                            samples)[0] - float(penalty)
    return {
        "captain": captain,
        "vice": vice,
        "p_beat_target": (p_beat_bar(mine, bar) if bar is not None
                          else p_beat_target(mine, rival_scores, target)),
        "rank_percentile": rank_percentile(mine, rival_scores),
        "mean_points": float(mine.mean()),
        "sd_points": float(mine.std()),
    }


def pick_best_squad(candidates: list, ids: list[int], samples: np.ndarray,
                    rival_scores: np.ndarray, target: float = 0.5,
                    bar: np.ndarray | None = None, penalties=None,
                    lineups=None, played: np.ndarray | None = None,
                    positions=None):
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
    costs = list(penalties) if penalties is not None else [0.0] * len(candidates)
    decisions = list(lineups) if lineups is not None else [None] * len(candidates)
    if len(decisions) != len(candidates):
        raise ValueError("one lineup is required for each candidate")
    scored = [score_candidate(c, ids, samples, rival_scores, target, bar=bar,
                              penalty=float(costs[i]), lineup=decisions[i],
                              played=played, positions=positions)
              for i, c in enumerate(candidates)]
    best = max(range(len(candidates)),
               key=lambda i: (scored[i]["p_beat_target"], scored[i]["mean_points"]))
    return candidates[best], scored


# --- resolving how good the field's BEST managers are ----------------------
# `RIVALS` is enough to locate the median manager and useless for the tail.
# Estimating the 99.99th percentile from 400 draws returns the maximum of 400,
# which came out 12 points low with an SD of 5.6 against a true bar of 111 --
# and "top of FPL" IS the 99.99th percentile, so the target that matters most
# was the one the sampler could not represent.
MIN_RIVALS = RIVALS
# Above this the draw costs more than the answer is worth. 0.9999 needs a
# million rivals; refusing is better than returning a number that is wrong by
# ten points without saying so.
MAX_RIVALS = 200_000
# Rival managers wanted ABOVE the bar before the bar is trusted. The K-th
# largest of R draws has a relative standard error near 1/sqrt(K), so 100 puts
# it around 10%.
RIVALS_ABOVE_BAR = 100
RIVAL_CHUNK = 2_000


def required_rivals(target: float) -> int:
    """How many rivals are needed to locate the `target` quantile.

    A quantile is only as well determined as the number of observations beyond
    it, so this scales as 1/(1 - target). Refuses rather than guesses when the
    answer would need more rivals than `MAX_RIVALS`.
    """
    target = float(target)
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target}")
    need = int(np.ceil(RIVALS_ABOVE_BAR / (1.0 - target)))
    if need > MAX_RIVALS:
        raise ValueError(
            f"a target of {target} cannot be resolved: it would need {need:,} "
            f"rival managers to place the bar and the cap is {MAX_RIVALS:,}. "
            f"The most extreme target this can measure is "
            f"{1 - RIVALS_ABOVE_BAR / MAX_RIVALS:.5f}."
        )
    return max(MIN_RIVALS, need)


def field_bar(xp_df: pd.DataFrame, samples: np.ndarray, target: float,
              rng: np.random.Generator, n_rivals: int | None = None) -> np.ndarray:
    """Per-simulation score the field's `target` quantile manager achieves.

    Drawn in chunks, keeping only the running top-K scores per simulation --
    the bar is the K-th largest, so nothing below it ever has to be stored.
    That is what makes a 200,000-rival field affordable: the dense
    (rivals x simulations) matrix it would otherwise need is gigabytes, while
    the top-K buffer is (K x simulations) with K around a hundred.
    """
    n_rivals = int(n_rivals) if n_rivals else required_rivals(target)
    n_sims = samples.shape[1]
    k = max(1, int(np.ceil((1.0 - float(target)) * n_rivals)))
    top = np.full((k, n_sims), -np.inf)
    drawn = 0
    while drawn < n_rivals:
        size = min(RIVAL_CHUNK, n_rivals - drawn)
        scores = squad_scores(sample_rival_squads(xp_df, size, rng), samples)
        merged = np.concatenate([top, scores], axis=0)
        # -k gives the k largest per column; their minimum is the bar.
        top = np.partition(merged, -k, axis=0)[-k:]
        drawn += size
    return top.min(axis=0)


def p_beat_bar(my_scores: np.ndarray, bar: np.ndarray) -> float:
    """P(this squad clears a per-simulation bar), as returned by `field_bar`."""
    return _clear_rate(np.asarray(my_scores, dtype=float), np.asarray(bar, dtype=float))
