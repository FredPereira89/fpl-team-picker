"""End-to-end weekly pipeline: data -> model -> optimize -> report.

Pure Python. No MCP, no skills, no third-party historical dataset - so a
headless cron invocation works by construction.
"""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .data.cache import (Cache, data_complete_after, is_matchday, final_through,
                         settled_after)
from .data.client import FplClient, DataCoverageError
from .data.normalize import (normalize_players, normalize_teams, normalize_fixtures,
                             history_past_frame, history_current_frame,
                             history_rounds_frame, apply_season_baseline,
                             latest_season)
from .data.store import save_table
from .data import snapshots
from .model.strength import team_ratings, league_goals_per_team_match
from .model.minutes import minutes_model
from .model.scoring import blended_rates
from .model.fixtures import team_fixture_frame, fixture_counts
from .model.xp import build_xp
from .backtest.ledger import save_predictions, load_scored_summary
from .model.calibration import fit_calibration, apply_calibration, scored_history
from .model.simulate import simulate_event_detailed, moment_match
from .optimize.squad import optimize_squad, enumerate_squads, Squad
from .optimize.rank import (sample_rival_squads, squad_scores, pick_best_squad,
                            field_bar, required_rivals, RIVALS, score_candidate,
                            lineup_scores)
from .optimize.lineup import build_lineup, best_xi
from .optimize.chips import advise_chips, ChipAdvice
from .optimize.actions import wildcard_action, freehit_action
from .optimize.transfers import (optimize_transfers, enumerate_transfer_plans,
                                 selling_price, bank_after)
from .optimize.multiperiod import optimize_multi_period
from .report.weekly import Recommendation, render

# Backtest result (scripts/run_backtest.py, trained on 2024/25, tested on
# 2025/26, run 2026-07-26): the shrunk per-90 baseline -- the entire model at
# GW1, since form_weight is 0 until GW6 -- beats both a naive last-season
# average and FPL's own published xP for DEF/MID/FWD (Spearman 0.34-0.60 vs
# 0.08-0.28), but shows no rank skill for goalkeepers (0.034, essentially
# uncorrelated, slightly below FPL's own xP). The backtest used a simplified
# single-rate proxy (shrunk points-per-90), not the exact production
# goal/assist/bonus/DC/saves component split, so the real GK figure may
# differ -- but there is no positive evidence for GK picks either way.
HORIZON_COL = "xp_horizon"

TRUST_SUMMARY = (
    "No gameweek of this model has been scored yet, so there is no measurement "
    "of it to quote. The only backtest that exists (2025/26 held out, trained "
    "on 2024/25, n=11406) measured a simplified points-per-90 PROXY, not the "
    "production goal/assist/bonus/DC/saves split -- outfield rank quality beat "
    "both a naive baseline and FPL's own published xP. Its goalkeeper verdict "
    "is deliberately not repeated here: it reported no rank skill, and the two "
    "gameweeks of the real model scored since (+0.524, +0.529) contradict it. "
    "Treat this run as unmeasured rather than trusted, and score a gameweek "
    "(scripts/score_gameweek.py) to replace this note with a real number."
)


def trust_text(root: Path) -> str:
    """What to tell the user about how far to trust the recommendation.

    A real scored gameweek always beats TRUST_SUMMARY, which describes a
    backtest of a simplified points-per-90 proxy rather than the production
    model -- and whose goalkeeper verdict the GW1 audit contradicted.
    """
    return load_scored_summary(root) or TRUST_SUMMARY


def freshness_flags(client, fixtures: list[dict], checked_through: int) -> list[str]:
    """Warn when this run's player history predates FPL's own data check.

    Only reachable for snapshots taken before `final_through` was recorded:
    anything newer either carries the marker and was verified, or was refused
    and re-fetched. The window is narrow -- between a gameweek's last whistle
    and FPL confirming bonus -- but a forecast built inside it is running on
    numbers that were still moving, and the report should say so rather than
    let the run look settled.
    """
    unverified = getattr(client, "unverified", set())
    if not unverified or checked_through <= 0:
        return []
    settled = settled_after(fixtures, checked_through)
    stamps = [v for k, v in getattr(client, "sources", {}).items() if k in unverified]
    if settled is None or not stamps:
        return []
    oldest = min(stamps)
    if oldest >= settled.strftime("%Y-%m-%dT%H:%M:%SZ"):
        return []
    return [
        f"Player history was cached at {oldest}, before GW{checked_through} was "
        f"confirmed by FPL, and is too old to record whether it was. Bonus and "
        f"stat corrections since then are not in this forecast — delete "
        f"data/cache/element-summary-*.json to force a refresh."
    ]


# How much of the player pool must have readable history before a run is
# allowed to optimise. Not 100%: a handful of unreadable fringe players changes
# nothing, and refusing the run over them would make the tool unusable during
# ordinary API flakiness. Below this, though, the pool the solver is choosing
# from is not the pool it thinks it is.
MIN_HISTORY_COVERAGE = 0.99


def coverage_gate(players, summaries, fetch_failed, owned_ids,
                  min_coverage: float = MIN_HISTORY_COVERAGE) -> list[str]:
    """Blocking reasons why this run's data is too incomplete to optimise on.

    Individual element-summary failures used to be swallowed silently. The
    missing player's prior rows were then zeroed and routed to the same price
    prior as a genuine new signing, so a partial API outage could erase an
    established player's history while the optimizer still returned a
    confident, legal team. The only signal was a global `stale` boolean, which
    cannot say whether the gap touches the squad.

    Two things are refused rather than warned about: an OWNED player this run
    could not read -- there is no safe way to price a sale or a hit against an
    estimate nothing supports -- and a pool whose overall coverage has fallen
    below `min_coverage`.
    """
    from .data.normalize import history_status, HISTORY_FAILED

    status = history_status(players, summaries, fetch_failed).set_index("player_id")
    name_of = dict(zip(players["player_id"].astype(int), players["web_name"]))
    failed = {int(i) for i, row in status.iterrows()
              if row["history_status"] == HISTORY_FAILED}
    if not failed:
        return []

    reasons = []
    owned_missing = sorted(failed & {int(i) for i in (owned_ids or [])})
    if owned_missing:
        who = ", ".join(f"{name_of.get(i, i)} ({i})" for i in owned_missing)
        reasons.append(
            f"player history could not be read for {len(owned_missing)} player(s) "
            f"you own: {who}. They would be priced off their transfer fee like a "
            f"new signing, which is how a bad sale or a hit gets recommended. "
            f"Re-run once the API is responding."
        )

    total = max(1, len(status))
    coverage = 1.0 - len(failed) / total
    if coverage < float(min_coverage):
        reasons.append(
            f"player history coverage is {coverage:.1%}, below the "
            f"{float(min_coverage):.0%} floor: {len(failed)} of {total} "
            f"summaries could not be read, so the pool the solver is choosing "
            f"from is not the one it appears to be."
        )
    return reasons


def _rank_context(xp, players, rates, minutes, tfx, cfg, from_event):
    """Simulated player points and the field to judge candidates against."""
    detail = simulate_event_detailed(players, rates, minutes, tfx, from_event,
                                     n_sims=int(cfg.rank_sims))
    ids, samples, played = detail["ids"], detail["samples"], detail["played"]
    # The MILP chose on CALIBRATED projections; the samples were drawn from raw
    # rates. Matching the means is what makes the rank layer score the squad
    # the solver actually proposed rather than an uncalibrated cousin of it.
    samples = moment_match(samples, ids, xp)
    rng = np.random.default_rng(0)
    rivals = sample_rival_squads(xp, n_rivals=RIVALS, rng=rng)
    rival_scores = squad_scores(rivals, samples)
    target = float(cfg.rank_target)
    n_needed = required_rivals(target)
    bar = (field_bar(xp, samples, target, rng, n_rivals=n_needed)
           if n_needed > RIVALS else None)
    positions = xp.set_index("player_id")["position"].astype(str).to_dict()
    return ids, samples, played, positions, rival_scores, target, n_needed, bar


def _p_gain_positive(chosen, plans, lineups, ids, samples, played,
                     positions) -> float | None:
    """P(this week's net gain over holding is positive), from the scenarios.

    A transfer is irreversible and a hit is a fixed cost, so the expected gain
    alone is the wrong summary: a +0.4 with a thin, uncertain edge and a +0.4
    on a nailed regular are not the same bet. This is a diagnostic for the
    report, not a selection rule -- the audit's caution about subtracting an
    "uncertainty penalty" from every low-confidence player applies.

    This does NOT yet carry the epistemic spread the difference between a
    price-prior newcomer and a nailed regular deserves: the simulation draws
    from the model's OWN point estimate of each player's rates and minutes,
    with no uncertainty layered on top of a single gameweek's draw (see R4 --
    a single-event marginal cannot be widened by construction; a real fix
    needs uncertainty reused across a multi-week horizon, which is unbuilt).
    A confidence-sensitive version of this number remains open work.
    """
    hold = next((p for p in plans if p.n_transfers == 0), None)
    if hold is None or chosen is hold or chosen.n_transfers == 0:
        return None

    def week(plan):
        lineup = lineups[plans.index(plan)]
        return lineup_scores(lineup, ids, samples, played, positions)

    gain = week(chosen) - week(hold) - float(chosen.hit_cost)
    return float(np.mean(gain > 0) + 0.5 * np.mean(gain == 0))


def _honour_rank_captain(lineup, rank_stats, xp):
    """Report the armband the rank layer actually scored, when it decided --
    and re-score mean_points/sd_points/p_beat_target/rank_percentile for
    whichever captain ends up reported.

    `pick_best_squad` records its own optimal captain per candidate, and the
    pipeline used to discard it and let `build_lineup` choose a mean-based one
    -- so the displayed team was not the team whose P(beat target) selected
    it. When the rank layer made the decision (a Mode 1 squad choice, or
    transfers with `rank_transfers` on), its captain is reported, provided he
    is in the exact one-week XI; otherwise the lineup's own choice stands and
    `rank_stats["captain_reported"]` says so. When rank only reported, the
    lineup's captain is the decision and nothing changes.

    Either way, the four scored statistics were computed under RANK'S OWN
    preferred captain (`score_candidate` re-optimises the armband per
    candidate). When that captain is not the one ending up reported --
    expected-points mode keeping its own captain, or a rank captain rejected
    for sitting outside the exact XI -- the stored numbers described a week
    that was never fielded (C6-2 fixed the XI; the armband could still
    diverge). `rank_stats` carries a private `_ctx` (the samples/rival field/
    bar this candidate was scored against) exactly so this function can
    re-score the FINAL captain here rather than leave rank's captain's
    numbers attached to someone else's armband. `_ctx` is stripped from the
    returned stats before they reach the report or get serialised.
    """
    if not rank_stats:
        return lineup, rank_stats
    decided = rank_stats.get("decided_by", "rank") == "rank"
    preferred_captain = rank_stats.get("captain")
    preferred_vice = rank_stats.get("vice")
    stats = dict(rank_stats)
    ctx = stats.pop("_ctx", None)
    xi = set(lineup.xi)
    if (decided and preferred_captain is not None
            and int(preferred_captain) in xi and preferred_vice is None):
        frame = xp.set_index("player_id")
        others = [pid for pid in lineup.xi if pid != int(preferred_captain)]
        preferred_vice = max(
            others, key=lambda pid: (float(frame.loc[pid, "xp_next1"]), -pid))
    honoured = (
        decided
        and preferred_captain is not None
        and int(preferred_captain) in xi
        and preferred_vice is not None
        and int(preferred_vice) in xi
        and int(preferred_captain) != int(preferred_vice)
    )
    if not honoured:
        stats["captain_reported"] = False
    else:
        captain, vice = int(preferred_captain), int(preferred_vice)
        if captain != lineup.captain or vice != lineup.vice:
            from dataclasses import replace
            frame = xp.set_index("player_id")
            # Lineup.xp = sum(XI xp) + the captain's own xp again (the armband
            # double). Overriding the captain without recomputing this left
            # the report's headline total describing the OLD captain: the
            # number shown and the armband shown belonged to two different
            # decisions.
            xi_total = sum(float(frame.loc[i, "xp_next1"]) for i in lineup.xi)
            new_xp = round(xi_total + float(frame.loc[captain, "xp_next1"]), 3)
            lineup = replace(lineup, captain=captain, vice=vice, xp=new_xp)
        stats["captain_reported"] = True
    ctx_ids = (set(ctx["lineup"].xi) if ctx is not None and "lineup" in ctx
               else set(ctx.get("starting_ids", ())) if ctx is not None else set())
    if ctx is not None and lineup.captain in ctx_ids:
        from types import SimpleNamespace
        score_args = dict(
            target=ctx["target"], bar=ctx["bar"], penalty=ctx["penalty"],
            captain=lineup.captain,
        )
        if "played" in ctx and "positions" in ctx:
            score_args.update(
                vice=lineup.vice, lineup=lineup, played=ctx["played"],
                positions=ctx["positions"],
            )
        rescored = score_candidate(
            SimpleNamespace(starting_ids=lineup.xi), ctx["ids"], ctx["samples"],
            ctx["rival_scores"], **score_args)
        stats["mean_points"] = rescored["mean_points"]
        stats["sd_points"] = rescored["sd_points"]
        stats["p_beat_target"] = rescored["p_beat_target"]
        stats["rank_percentile"] = rescored["rank_percentile"]
    # Keep the alternative armband as explicitly labelled diagnostic metadata;
    # `captain`/`vice` always identify the decision the stored statistics score.
    stats["rank_preferred_captain"] = preferred_captain
    stats["rank_preferred_vice"] = preferred_vice
    stats["captain"] = int(lineup.captain)
    stats["vice"] = int(lineup.vice)
    return lineup, stats


def _rank_stats(scored, index, n_candidates, target, n_rivals) -> dict:
    stats = dict(scored[index])
    stats["n_candidates"] = int(n_candidates)
    stats["target"] = float(target)
    stats["n_rivals"] = int(n_rivals)
    return stats


def _stash_rank_context(stats, ids, samples, played, positions, rival_scores,
                        bar, target, lineup, penalty=0.0) -> dict:
    """Attach what `_honour_rank_captain` needs to re-score the FINAL
    decision later, once `build_lineup` has settled what is reported. Chip
    overrides that replace the squad clear rank stats separately. Private to
    this module -- `_honour_rank_captain` pops `_ctx` before the stats reach
    the report or get serialised.
    """
    stats["_ctx"] = {
        "ids": ids, "samples": samples, "rival_scores": rival_scores,
        "played": played, "positions": positions, "bar": bar,
        "target": float(target), "lineup": lineup, "penalty": float(penalty),
    }
    return stats


def _last_event(bootstrap: dict, fixtures) -> int:
    """The final gameweek of the season.

    Bootstrap is authoritative when it carries an event list; a fixture frame
    is the fallback, and 38 the last resort. Chip patience is measured against
    this, so getting it wrong makes the advisor either too precious or too
    hasty at the end of a season.
    """
    events = [int(e["id"]) for e in bootstrap.get("events", []) if e.get("id")]
    if events:
        return max(events)
    ev = pd.to_numeric(fixtures["event"], errors="coerce").dropna()
    return int(ev.max()) if len(ev) else 38


def _squad_quality(xp, cfg, current_squad, bank, selling, best_plan):
    """How far the squad sits below what its own money could buy.

    A Wildcard is unlimited transfers, which is exactly `_solve` with every one
    of the fifteen in play -- same objective, same budget, same selling prices
    as the weekly plans, so the two totals are directly comparable. Routing
    this through optimize_squad instead would have compared against a fresh
    cfg.budget the manager does not have.

    The surplus is net of `best_plan.gain`: a squad one good transfer away from
    its own optimum has nothing here for the chip to buy.
    """
    from .optimize.transfers import _budget_and_cost, _plan, _solve
    from .optimize.chips import SquadQuality

    current = {int(i) for i in current_squad}
    budget, cost = _budget_and_cost(xp, current, bank, selling)
    # A wildcard really does rebuild all fifteen, so it is measured against the
    # squad the BUILDER would produce -- bench floor included -- not against one
    # the weekly optimizer would never pick.
    solved = _solve(xp, current, budget, len(current), cfg, HORIZON_COL, cost=cost,
                    bench_floor=float(getattr(cfg, "bench_floor_xp", 0.0) or 0.0))
    if solved is None:   # floor unaffordable on this budget -- measure without it
        solved = _solve(xp, current, budget, len(current), cfg, HORIZON_COL, cost=cost)
    if solved is None:
        return None
    rebuild = _plan(current, solved, len(current), cfg)
    # The hold baseline is already solved -- every enumerated plan carries it as
    # baseline_xp -- so this costs one solve, not two.
    surplus = (rebuild.net_xp - float(best_plan.baseline_xp)) - float(best_plan.gain)
    return SquadQuality(
        surplus=round(surplus, 3),
        changes=int(rebuild.n_transfers),
        hit_equivalent=float(rebuild.n_transfers * int(cfg.hit_cost)),
    )


def _with_weekly_xi(candidates, xp):
    """Replace each candidate's `starting_ids` with the EXACT one-week XI.

    Every candidate out of `enumerate_squads`/`enumerate_transfer_plans`
    still carries the solver's discounted-HORIZON XI -- the one it holds
    fixed for the whole projection window (B8's known limitation). Rank
    scoring (`score_candidate`, `best_captain_by_rank`) and the gain
    diagnostic (`_p_gain_positive`) used to run on that horizon XI, then the
    pipeline re-picked the exact weekly XI only AFTERWARDS for the report --
    so the default rank diagnostics described a different lineup from the
    one shown, the rank-preferred captain could be a player benched in the
    exact XI, and an opt-in `rank_squad`/`rank_transfers` could decide on a
    lineup that was never actually fielded.

    `best_xi` is a pure function of the fifteen and the projection column, so
    substituting it here -- ONCE, before any scoring happens -- makes every
    later recompute (the final `build_lineup` call) agree with it exactly;
    there is no way for the two to drift apart afterwards. `net_xp`/
    `gross_xp`/`gain` (the horizon totals reported as "suggested net gain") are
    untouched: those describe the multi-gameweek decision the solver actually
    made and do not depend on which single week's XI is being displayed.
    """
    from dataclasses import replace
    out = []
    for c in candidates:
        full = list(getattr(c, "squad_ids", None) or c.player_ids)
        out.append(replace(c, starting_ids=best_xi(full, xp, xp_col="xp_next1")))
    return out


def _candidate_lineups(candidates, xp):
    """Build the complete production decision for each rank candidate.

    `_with_weekly_xi` has already selected the exact XI.  `exact=False` keeps
    that XI while deriving the same bench order and risk-aware captain/vice
    pair the final report uses.  These objects are then shared by rank scoring,
    gain diagnostics, and the final reconciliation step.
    """
    out = []
    for candidate in candidates:
        full = list(getattr(candidate, "squad_ids", None) or candidate.player_ids)
        out.append(build_lineup(
            Squad(full, list(candidate.starting_ids), 0.0, 0.0), xp,
            exact=False,
        ))
    return out


def _choose_transfers(xp, players, rates, minutes, tfx, cfg, from_event,
                      current_squad, bank, free_transfers, selling):
    """This week's transfer plan, chosen on discounted multi-gameweek net points.

    The rank layer used to make this decision, and it should not. Candidates are
    generated on the discounted horizon, but the final pick was made purely on
    CURRENT-EVENT samples: a plan's future gains were never scored at all. A
    transfer losing 0.2 this week and gaining 8 over the next four lost to a
    one-week move, and a hit with strong future payback was close to
    unselectable -- which defeats the entire purpose of charging four points
    for it. When rank simulation was enabled by default, that one-week
    objective superseded the otherwise-correct horizon comparison every
    gameweek.

    So the horizon decides, and the rank layer reports. `optimizer.rank_transfers`
    restores the old behaviour for anyone who wants it, and is off by default
    because a one-week target cannot price a five-week decision.
    """
    rolling = None
    if bool(getattr(cfg, "multi_period_transfers", False)):
        rolling = optimize_multi_period(
            xp, current_squad, bank, free_transfers, cfg,
            selling_prices=selling).first_week_plan()

    plans = []
    lineups = []
    if int(cfg.rank_sims) > 0:
        # Enumerated even when rank does not decide: the candidate set is how a
        # hit and a coordinated two-move restructure get onto the table at all,
        # and the diversity is worth having in the report.
        plans = enumerate_transfer_plans(xp, current_squad, bank, free_transfers, cfg,
                                         xp_col=HORIZON_COL, selling_prices=selling,
                                         k=int(cfg.rank_candidates))
        if rolling is not None:
            duplicate = next((i for i, p in enumerate(plans)
                              if set(p.squad_ids) == set(rolling.squad_ids)), None)
            if duplicate is None:
                plans.append(rolling)
            else:
                plans[duplicate] = rolling
        if plans:
            plans = _with_weekly_xi(plans, xp)
            lineups = _candidate_lineups(plans, xp)
            rolling = next(
                (p for p in plans if p.strategy == "multi-period"), rolling)

    if plans and int(cfg.rank_sims) > 0 and bool(getattr(cfg, "rank_transfers", False)):
        (ids, samples, played, positions, rival_scores,
         target, n_needed, bar) = _rank_context(
             xp, players, rates, minutes, tfx, cfg, from_event)
        chosen, scored = pick_best_squad(plans, ids, samples, rival_scores, target=target,
                                         bar=bar, penalties=[p.hit_cost for p in plans],
                                         lineups=lineups, played=played,
                                         positions=positions)
        index = plans.index(chosen)
        stats = _rank_stats(scored, index, len(plans), target, n_needed)
        stats["hit_cost"] = int(chosen.hit_cost)
        stats["decided_by"] = "rank"
        _stash_rank_context(stats, ids, samples, played, positions, rival_scores,
                           bar, target, lineups[index], penalty=chosen.hit_cost)
        return chosen, None, stats

    if plans:
        # net_xp is the discounted horizon total with this plan's hit already
        # subtracted, which is exactly the quantity a transfer decision turns on.
        best = rolling or max(plans, key=lambda p: (p.net_xp, -p.n_transfers))
        stats = None
        if int(cfg.rank_sims) > 0:
            # Diagnostic only: how the CHOSEN plan fares against the field. It
            # no longer selects anything, so it cannot overrule the horizon.
            (ids, samples, played, positions, rival_scores,
             target, n_needed, bar) = _rank_context(
                 xp, players, rates, minutes, tfx, cfg, from_event)
            _, scored = pick_best_squad(plans, ids, samples, rival_scores, target=target,
                                        bar=bar, penalties=[p.hit_cost for p in plans],
                                        lineups=lineups, played=played,
                                        positions=positions)
            best_index = plans.index(best)
            stats = _rank_stats(scored, best_index, len(plans), target, n_needed)
            stats["hit_cost"] = int(best.hit_cost)
            stats["decided_by"] = "expected points over the horizon"
            stats["p_gain_positive"] = _p_gain_positive(
                best, plans, lineups, ids, samples, played, positions)
            _stash_rank_context(stats, ids, samples, played, positions, rival_scores,
                               bar, target, lineups[best_index], penalty=best.hit_cost)
        return best, plans, stats

    if rolling is not None:
        return rolling, [rolling], None

    best, options = optimize_transfers(xp, current_squad, bank, free_transfers, cfg,
                                       xp_col=HORIZON_COL, selling_prices=selling)
    return best, options, None


def _choose_squad(xp, players, rates, minutes, tfx, cfg, from_event):
    """The squad to recommend, and how it fares against a simulated field.

    With `optimizer.rank_sims` at 0 this is the plain MILP optimum and nothing
    is simulated. Otherwise the solver proposes its best `rank_candidates`
    squads and the simulation picks between them by how often they beat the
    field -- the step that lets a correlated bet (three defenders, one clean
    sheet) be priced as the single bet it actually is, which no linear
    objective can do.
    """
    if int(cfg.rank_sims) <= 0:
        return optimize_squad(xp, cfg, xp_col=HORIZON_COL), None

    candidates = enumerate_squads(xp, cfg, xp_col=HORIZON_COL,
                                  k=int(cfg.rank_candidates),
                                  min_different=int(cfg.rank_diversity))
    if not candidates:
        return optimize_squad(xp, cfg, xp_col=HORIZON_COL), None
    candidates = _with_weekly_xi(candidates, xp)
    lineups = _candidate_lineups(candidates, xp)

    # The bar for the configured target is drawn from as many rivals as that
    # target needs -- 400 cannot locate anything past about the 99th
    # percentile, and "top of FPL" lives far beyond it.
    (ids, samples, played, positions, rival_scores,
     target, n_needed, bar) = _rank_context(
         xp, players, rates, minutes, tfx, cfg, from_event)
    chosen, scored = pick_best_squad(candidates, ids, samples, rival_scores,
                                     target=target, bar=bar, lineups=lineups,
                                     played=played, positions=positions)
    if bool(getattr(cfg, "rank_squad", False)):
        chosen_index = candidates.index(chosen)
        stats = _rank_stats(scored, chosen_index, len(candidates), target, n_needed)
        stats["decided_by"] = "rank"
        _stash_rank_context(stats, ids, samples, played, positions, rival_scores,
                           bar, target, lineups[chosen_index])
        return chosen, stats
    # Expected points decide -- candidates[0] is the solver's optimum -- and
    # the rank layer only reports how that squad fares. A one-week
    # median-beat probability is neither expected points nor expected rank,
    # and letting it overrule the objective traded mean for the wrong kind of
    # variance every week by default.
    best = candidates[0]
    stats = _rank_stats(scored, 0, len(candidates), target, n_needed)
    stats["decided_by"] = "expected points over the horizon"
    stats["rank_would_choose"] = candidates.index(chosen)
    _stash_rank_context(stats, ids, samples, played, positions, rival_scores,
                       bar, target, lineups[0])
    return best, stats


def run(cfg: Config, mode: int, from_event: int, root: Path, client=None,
        news=None, current_squad=None, bank: float = 0.0, free_transfers: int = 1,
        progress=None, purchase_prices: dict[int, float] | None = None,
        chip_events: list[dict] | None = None, first_event: int = 1):
    root = Path(root)
    client = client or FplClient(Cache(root / "cache"), ttl_hours=cfg.cache_ttl_hours)

    # Fixtures first, because whether today is a matchday decides how long
    # everything else may be cached. On a matchday prices, news and status
    # change hour to hour, and a six-hour-old bootstrap can have a player still
    # listed as available who was ruled out at the team-sheet announcement.
    raw_fixtures = client.fixtures()
    if is_matchday(raw_fixtures, datetime.now(timezone.utc)):
        ttl = float(getattr(client, "ttl_hours", cfg.cache_ttl_hours))
        client.ttl_hours = min(ttl, float(cfg.cache_ttl_matchday_hours))
    bootstrap = client.bootstrap()
    players = normalize_players(bootstrap)
    teams = normalize_teams(bootstrap)
    fixtures = normalize_fixtures(raw_fixtures)

    # bootstrap-static's counting stats are CURRENT-season cumulative and get
    # reset to zero at the season rollover, but the model reads them as a full
    # season of history (per90_rates shrinks with k=900 minutes; minutes_model
    # divides starts by 38). Before GW1 those totals still show last season and
    # the model works; from GW1 they show a handful of games and every
    # established player collapses to a tiny sample. Re-source the baseline from
    # element-summary history_past, which is stable all season. Pre-season this
    # is a no-op -- the two agree -- so it runs unconditionally rather than on a
    # brittle "has the season started" test.
    # These summaries carry BOTH the stable `history_past` baseline and this
    # season's `history` (read below by history_current_frame). The 30-day TTL
    # is right for the former and silently wrong for the latter, so the cache
    # is additionally required to postdate the last finished match -- otherwise
    # a snapshot taken in GW1 keeps counting as fresh into December and the
    # form blend, which ramps with gws_played, stays pinned at its GW1 weight.
    # ...and freshness alone is not enough either. `data_complete_after` asks
    # whether the football has been played; FPL's bonus and stat check lands
    # after that, and a snapshot taken in between carries numbers that are
    # still moving. Snapshots now record which gameweeks FPL had checked when
    # they were taken, and one that predates the check is refused.
    checked_through = final_through(raw_fixtures)
    if hasattr(client, "snapshot_meta"):
        client.snapshot_meta = {"final_through": checked_through}
    summaries = client.element_summaries(players["player_id"].tolist(), progress=progress,
                                         not_before=data_complete_after(raw_fixtures),
                                         require_final_through=checked_through)
    # Refuse to optimise on a pool this run could not actually read. A missing
    # summary is indistinguishable downstream from a genuine newcomer, so an
    # outage would otherwise produce a confident team built on price priors.
    blocking = coverage_gate(players, summaries,
                             getattr(client, "fetch_failures", set()),
                             owned_ids=current_squad or [])
    if blocking:
        raise DataCoverageError(" ".join(blocking))

    past = history_past_frame(summaries)
    baseline_season = latest_season(past)
    players = apply_season_baseline(players, past, baseline_season)

    processed = root / "processed"
    save_table(players, "players", processed)
    save_table(teams, "teams", processed)
    save_table(fixtures, "fixtures", processed)

    # Season-to-date output and starts, strictly before the gameweek being
    # predicted. Until 2026-08-27 nothing read this: the model ran entirely on
    # last season and could not see the current one, which is also why a summer
    # signing with no prior Premier League row stayed pinned to the positional
    # mean however he played.
    current = history_current_frame(summaries, before_event=from_event)

    ratings = team_ratings(players, teams, current=current)
    # team_ratings returns ratios centred on 1.0; the fixture model needs a real
    # goals-per-match rate to turn them into expected goals conceded, or every
    # clean-sheet probability comes out ~0.44 against a true rate near 0.27.
    league_gc = league_goals_per_team_match(players)
    tfx = team_fixture_frame(fixtures, ratings, from_event, cfg.horizon_gw,
                             league_gc=league_gc)
    # Fixture STRUCTURE for the whole rest of the season, not just the
    # projection horizon. Chip timing is a season-long decision -- holding a
    # Bench Boost for a GW29 double is the entire point of having one -- and
    # counting fixtures stays cheap in gameweeks where projecting points does
    # not. model.xp never used this frame; only the chip advisor does.
    last_event = _last_event(bootstrap, fixtures)
    counts = fixture_counts(fixtures, list(teams["team_id"]), from_event,
                            max(1, last_event - from_event + 1))
    # The same history match by match. Totals cannot say WHEN the output came,
    # how often a start lasted the hour, or how long a start lasts -- and all
    # three were being answered with constants.
    rounds = history_rounds_frame(summaries, before_event=from_event)
    rates = blended_rates(players, current, cfg, rounds=rounds)
    minutes = minutes_model(players, cfg, news=news, current=current, rounds=rounds)
    xp = build_xp(players, rates, minutes, tfx, cfg, from_event)

    # Recalibrate per position against gameweeks already scored. Position is
    # the axis that matters: a GLOBAL affine correction cannot change any pick,
    # because every squad has the same fifteen slots and the argmax is
    # unmoved by shifting and scaling every candidate identically.
    calibration_note = None
    if getattr(cfg, "calibrate", True):
        history = scored_history(root, summaries, from_event)
        cal = fit_calibration(history)
        if cal is not None:
            xp = apply_calibration(xp, cal,
                                   decay=float(getattr(cfg, "horizon_decay", 1.0)))
            calibration_note = cal.summary

    # Record the forecast before acting on it. Scoring it later (fpl.backtest.
    # ledger, scripts/score_gameweek.py) is the only thing that measures the
    # production model rather than a proxy of it.
    #
    # The deadline goes in with it: without one the ledger cannot tell a
    # pre-deadline forecast, which the manager could actually have acted on,
    # from a post-deadline re-run that already knows the team news.
    deadline = next(
        (e["deadline_time"] for e in bootstrap.get("events", [])
         if e["id"] == from_event), None)
    # And record what this run could SEE, so a later replay of this gameweek
    # does not have to read today's prices, availability and club assignments.
    # Every capture is kept, and the forecast records which one it read.
    from dataclasses import asdict, is_dataclass
    snapshot = snapshots.capture(
        root, from_event, bootstrap=bootstrap, fixtures=raw_fixtures,
        deadline=deadline, final_through=checked_through,
        sources=client.source_summary() if hasattr(client, "source_summary") else {},
        news=news or {}, config=asdict(cfg) if is_dataclass(cfg) else dict(cfg))
    save_predictions(xp, from_event, root, cfg=cfg,
                     sources=client.source_summary()
                     if hasattr(client, "source_summary") else {},
                     deadline=deadline, snapshot=snapshot)

    # actual_mode reflects which branch genuinely ran, not the caller's
    # request -- Mode 2 needs a current_squad to transfer from, and nothing
    # in this codebase fetches one yet, so a mode=2 call with no
    # current_squad must be labelled and reported as the Mode 1 rebuild it
    # actually is, never silently mislabelled as a transfer recommendation.
    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    transfers = None
    rank_stats = None
    selling = {}
    if mode == 2 and current_squad:
        actual_mode = 2
        # What FPL would actually pay for each owned player. Falls back to market
        # price for anyone whose purchase price was never recorded, which is the
        # honest default -- it can only overstate, never understate, the budget.
        selling = {
            int(i): selling_price((purchase_prices or {}).get(int(i), prices[int(i)]),
                                  prices[int(i)])
            for i in current_squad
        }
        # xp_horizon, not xp_next5: the solver should discount gains it may
        # never collect, while the report still shows the honest raw total.
        best, _options, rank_stats = _choose_transfers(
            xp, players, rates, minutes, tfx, cfg, from_event,
            current_squad, bank, free_transfers, selling)
        squad_ids, starting_ids, transfers = best.squad_ids, best.starting_ids, best
        # R8 deliberately does not optimise chips yet. Its objective includes
        # future transfers and terminal FT value, while `_squad_quality` values
        # a fixed post-Wildcard squad; subtracting those unlike currencies can
        # manufacture or hide a Wildcard surplus. Leave that one heuristic off
        # until chips are modelled inside the rolling path.
        quality = (None if best.strategy == "multi-period" else
                   _squad_quality(xp, cfg, current_squad, bank, selling, best))
    else:
        actual_mode = 1
        quality = None   # no existing squad to have drifted
        squad, rank_stats = _choose_squad(xp, players, rates, minutes, tfx, cfg,
                                          from_event)
        squad_ids, starting_ids = squad.player_ids, squad.starting_ids

    lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)
    lineup, rank_stats = _honour_rank_captain(lineup, rank_stats, xp)

    team_by_player = dict(zip(players["player_id"].astype(int), players["team_id"].astype(int)))
    # Chips already spent, DATED, from local state merged with FPL's own record.
    # This was a literal [] until 2026-09-07, so the advisor happily recommended
    # a Wildcard that had been played weeks earlier -- and a bare name list then
    # suppressed the second-half copy of every chip for the rest of the season.
    chip = advise_chips(xp, lineup, squad_ids, counts, team_by_player, from_event,
                        list(chip_events or []), last_event=last_event,
                        quality=quality, first_event=first_event)

    # A named chip must return the squad that chip actually fields. Advising
    # "Free Hit" while handing back the ordinary permanent transfer plan is not
    # a timing error -- it is not the action the chip means.
    chip_squad = chip_temporary = False
    if actual_mode == 2 and chip is not None and chip.chip in ("wildcard", "freehit"):
        action = (wildcard_action(xp, cfg, current_squad, bank, selling,
                                  xp_col=HORIZON_COL)
                  if chip.chip == "wildcard"
                  else freehit_action(xp, cfg, current_squad, bank, selling))
        if action is None:
            chip = ChipAdvice(None, (
                f"A {chip.chip} looked right this week, but no legal squad could "
                f"be built from your budget — so no chip is recommended."
            ))
        else:
            squad_ids, starting_ids = action.squad_ids, action.starting_ids
            transfers = action.transfers
            lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)
            chip_squad, chip_temporary = True, action.temporary
            # rank_stats above described the ORDINARY transfer plan's squad --
            # a wildcard/free hit here replaces it with an entirely different
            # fifteen, not merely a different captain, so those numbers no
            # longer describe anything about the team now being reported.
            # Recomputing them would mean a second full rank simulation for a
            # squad the ordinary candidate pool never considered; clearing
            # them is the honest alternative to leaving stale figures attached
            # to a squad they were never scored against.
            rank_stats = None

    value = round(sum(prices[i] for i in squad_ids), 1)

    if actual_mode == 2:
        # Bank must reflect proceeds from the CURRENT squad, not the new one,
        # at selling value rather than market value (see optimize.transfers).
        cash = bank_after(bank, current_squad, squad_ids, prices, purchase_prices)
    else:
        cash = round(cfg.budget - value, 1)

    rec = Recommendation(
        rank=rank_stats,
        calibration=calibration_note,
        gw=from_event, deadline=deadline or "see the FPL site",
        mode=actual_mode, lineup=lineup,
        squad_ids=squad_ids, transfers=transfers, chip=chip,
        flags=freshness_flags(client, raw_fixtures, checked_through),
        bank=cash,
        squad_value=value, stale=getattr(client, "stale", False),
        chip_squad=chip_squad, chip_temporary=chip_temporary,
        trust=trust_text(root) if actual_mode == 1 else "",
    )
    return rec, xp
