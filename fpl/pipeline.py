"""End-to-end weekly pipeline: data -> model -> optimize -> report.

Pure Python. No MCP, no skills, no third-party historical dataset - so a
headless cron invocation works by construction.
"""
from pathlib import Path
import pandas as pd

from .config import Config
from .data.cache import Cache
from .data.client import FplClient
from .data.normalize import (normalize_players, normalize_teams, normalize_fixtures,
                             history_past_frame, apply_season_baseline, latest_season)
from .data.store import save_table
from .model.strength import team_ratings, league_goals_per_team_match
from .model.minutes import minutes_model
from .model.scoring import per90_rates
from .model.fixtures import team_fixture_frame, fixture_counts
from .model.xp import build_xp
from .optimize.squad import optimize_squad
from .optimize.lineup import build_lineup
from .optimize.chips import advise_chips
from .optimize.transfers import optimize_transfers
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
TRUST_SUMMARY = (
    "Backtest complete (2025/26 held out, trained on 2024/25, n=11406 GW "
    "observations): outfield rank quality (DEF/MID/FWD) beats both a naive "
    "last-season baseline and FPL's own published xP -- treat those picks "
    "with normal confidence. Goalkeeper rank quality shows no measurable "
    "skill (Spearman 0.034) and is not shown to beat FPL's own xP -- treat "
    "GK picks with extra caution; consider leaning on FPL's own projections "
    "or team news for that position specifically."
)


def run(cfg: Config, mode: int, from_event: int, root: Path, client=None,
        news=None, current_squad=None, bank: float = 0.0, free_transfers: int = 1,
        progress=None):
    root = Path(root)
    client = client or FplClient(Cache(root / "cache"), ttl_hours=cfg.cache_ttl_hours)

    bootstrap = client.bootstrap()
    raw_fixtures = client.fixtures()
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
    summaries = client.element_summaries(players["player_id"].tolist(), progress=progress)
    past = history_past_frame(summaries)
    baseline_season = latest_season(past)
    players = apply_season_baseline(players, past, baseline_season)

    processed = root / "processed"
    save_table(players, "players", processed)
    save_table(teams, "teams", processed)
    save_table(fixtures, "fixtures", processed)

    ratings = team_ratings(players, teams)
    # team_ratings returns ratios centred on 1.0; the fixture model needs a real
    # goals-per-match rate to turn them into expected goals conceded, or every
    # clean-sheet probability comes out ~0.44 against a true rate near 0.27.
    league_gc = league_goals_per_team_match(players)
    tfx = team_fixture_frame(fixtures, ratings, from_event, cfg.horizon_gw,
                             league_gc=league_gc)
    counts = fixture_counts(fixtures, list(teams["team_id"]), from_event, cfg.horizon_gw)
    rates = per90_rates(players, cfg)
    minutes = minutes_model(players, cfg, news=news)
    xp = build_xp(players, rates, minutes, tfx, counts, cfg, from_event)

    # actual_mode reflects which branch genuinely ran, not the caller's
    # request -- Mode 2 needs a current_squad to transfer from, and nothing
    # in this codebase fetches one yet, so a mode=2 call with no
    # current_squad must be labelled and reported as the Mode 1 rebuild it
    # actually is, never silently mislabelled as a transfer recommendation.
    transfers = None
    if mode == 2 and current_squad:
        actual_mode = 2
        best, _options = optimize_transfers(xp, current_squad, bank, free_transfers, cfg)
        squad_ids, starting_ids, transfers = best.squad_ids, best.starting_ids, best
    else:
        actual_mode = 1
        squad = optimize_squad(xp, cfg)
        squad_ids, starting_ids = squad.player_ids, squad.starting_ids

    from .optimize.squad import Squad
    lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)

    team_by_player = dict(zip(players["player_id"].astype(int), players["team_id"].astype(int)))
    chip = advise_chips(xp, lineup, squad_ids, counts, team_by_player, from_event, [])

    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    value = round(sum(prices[i] for i in squad_ids), 1)
    deadline = next(
        (e["deadline_time"] for e in bootstrap.get("events", []) if e["id"] == from_event),
        "see the FPL site",
    )

    if actual_mode == 2:
        # Bank must reflect proceeds from the CURRENT squad, not the new one --
        # optimize_transfers spends bank + value(current_squad), so recompute
        # against the pre-transfer value rather than passing the caller's
        # pre-transfer bank straight through unchanged.
        value_before = round(sum(prices[i] for i in current_squad), 1)
        bank_after = round(bank + value_before - value, 1)
    else:
        bank_after = round(cfg.budget - value, 1)

    rec = Recommendation(
        gw=from_event, deadline=deadline, mode=actual_mode, lineup=lineup,
        squad_ids=squad_ids, transfers=transfers, chip=chip,
        bank=bank_after,
        squad_value=value, stale=getattr(client, "stale", False),
        trust=TRUST_SUMMARY if actual_mode == 1 else "",
    )
    return rec, xp
