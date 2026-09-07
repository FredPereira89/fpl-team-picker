"""End-to-end weekly pipeline: data -> model -> optimize -> report.

Pure Python. No MCP, no skills, no third-party historical dataset - so a
headless cron invocation works by construction.
"""
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from .config import Config
from .data.cache import (Cache, data_complete_after, is_matchday, final_through,
                         settled_after)
from .data.client import FplClient
from .data.normalize import (normalize_players, normalize_teams, normalize_fixtures,
                             history_past_frame, history_current_frame,
                             history_rounds_frame, apply_season_baseline,
                             latest_season)
from .data.store import save_table
from .model.strength import team_ratings, league_goals_per_team_match
from .model.minutes import minutes_model
from .model.scoring import blended_rates
from .model.fixtures import team_fixture_frame, fixture_counts
from .model.xp import build_xp
from .backtest.ledger import save_predictions, load_scored_summary
from .optimize.squad import optimize_squad
from .optimize.lineup import build_lineup
from .optimize.chips import advise_chips
from .optimize.transfers import optimize_transfers, selling_price, bank_after
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
    "Backtest complete (2025/26 held out, trained on 2024/25, n=11406 GW "
    "observations): outfield rank quality (DEF/MID/FWD) beats both a naive "
    "last-season baseline and FPL's own published xP -- treat those picks "
    "with normal confidence. Goalkeeper rank quality shows no measurable "
    "skill (Spearman 0.034) and is not shown to beat FPL's own xP -- treat "
    "GK picks with extra caution; consider leaning on FPL's own projections "
    "or team news for that position specifically."
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


def run(cfg: Config, mode: int, from_event: int, root: Path, client=None,
        news=None, current_squad=None, bank: float = 0.0, free_transfers: int = 1,
        progress=None, purchase_prices: dict[int, float] | None = None,
        chips_used: list[str] | None = None):
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
    counts = fixture_counts(fixtures, list(teams["team_id"]), from_event, cfg.horizon_gw)
    # The same history match by match. Totals cannot say WHEN the output came,
    # how often a start lasted the hour, or how long a start lasts -- and all
    # three were being answered with constants.
    rounds = history_rounds_frame(summaries, before_event=from_event)
    rates = blended_rates(players, current, cfg, rounds=rounds)
    minutes = minutes_model(players, cfg, news=news, current=current, rounds=rounds)
    xp = build_xp(players, rates, minutes, tfx, counts, cfg, from_event)

    # Record the forecast before acting on it. Scoring it later (fpl.backtest.
    # ledger, scripts/score_gameweek.py) is the only thing that measures the
    # production model rather than a proxy of it.
    save_predictions(xp, from_event, root, cfg=cfg,
                     sources=client.source_summary()
                     if hasattr(client, "source_summary") else {})

    # actual_mode reflects which branch genuinely ran, not the caller's
    # request -- Mode 2 needs a current_squad to transfer from, and nothing
    # in this codebase fetches one yet, so a mode=2 call with no
    # current_squad must be labelled and reported as the Mode 1 rebuild it
    # actually is, never silently mislabelled as a transfer recommendation.
    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    transfers = None
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
        best, _options = optimize_transfers(xp, current_squad, bank, free_transfers, cfg,
                                            xp_col=HORIZON_COL, selling_prices=selling)
        squad_ids, starting_ids, transfers = best.squad_ids, best.starting_ids, best
    else:
        actual_mode = 1
        squad = optimize_squad(xp, cfg, xp_col=HORIZON_COL)
        squad_ids, starting_ids = squad.player_ids, squad.starting_ids

    from .optimize.squad import Squad
    lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)

    team_by_player = dict(zip(players["player_id"].astype(int), players["team_id"].astype(int)))
    # Chips already spent, from local state merged with FPL's own record. This
    # was a literal [] until 2026-09-07, so the advisor happily recommended a
    # Wildcard that had been played weeks earlier.
    chip = advise_chips(xp, lineup, squad_ids, counts, team_by_player, from_event,
                        list(chips_used or []))

    value = round(sum(prices[i] for i in squad_ids), 1)
    deadline = next(
        (e["deadline_time"] for e in bootstrap.get("events", []) if e["id"] == from_event),
        "see the FPL site",
    )

    if actual_mode == 2:
        # Bank must reflect proceeds from the CURRENT squad, not the new one,
        # at selling value rather than market value (see optimize.transfers).
        cash = bank_after(bank, current_squad, squad_ids, prices, purchase_prices)
    else:
        cash = round(cfg.budget - value, 1)

    rec = Recommendation(
        gw=from_event, deadline=deadline, mode=actual_mode, lineup=lineup,
        squad_ids=squad_ids, transfers=transfers, chip=chip,
        flags=freshness_flags(client, raw_fixtures, checked_through),
        bank=cash,
        squad_value=value, stale=getattr(client, "stale", False),
        trust=trust_text(root) if actual_mode == 1 else "",
    )
    return rec, xp
