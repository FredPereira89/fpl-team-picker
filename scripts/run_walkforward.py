#!/usr/bin/env python
"""Replay past gameweeks and report whether the model actually beats the field.

Populates the prediction ledger with a forecast for every gameweek already
played, built only from data that existed before it, then scores each one
against FPL's own published average. The ledger is what the per-position
calibration (model.calibration) fits on, so running this is what turns the
calibration from a no-op into a correction.

    python scripts/run_walkforward.py --through 8

Read the verdict, not the mean. The weekly edge has an SD near 15 points: ten
gameweeks can only confirm an edge of ~13 pts/GW, a full season ~6.8, and
reaching the top 1k of FPL needs ~12.6 sustained. `weekly_edge` says which of
those the sample can actually support.
"""
import argparse
import json
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl.config import load_config
from fpl.data.normalize import (normalize_players, normalize_teams, normalize_fixtures,
                                history_past_frame, apply_season_baseline, latest_season)
from fpl.model.strength import team_ratings, league_goals_per_team_match
from fpl.model.minutes import minutes_model
from fpl.model.scoring import blended_rates
from fpl.model.fixtures import team_fixture_frame
from fpl.model.xp import build_xp
from fpl.model.calibration import fit_calibration, apply_calibration, scored_history
from fpl.backtest.ledger import save_predictions
from fpl.backtest.walkforward import (forecast_inputs, actuals_frame, realised_score,
                                      squad_ledger, weekly_edge, replayable_gameweeks)
from fpl.optimize.squad import optimize_squad


def newest_cached(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no cached data matching {pattern} -- run the pipeline first")
    return json.load(open(files[-1], encoding="utf-8"))


def load_summaries(cache: Path) -> dict:
    best = {}
    for p in glob.glob(str(cache / "element-summary-*.json")):
        m = re.match(r"element-summary-(\d+)_(\d+T\d+Z)\.json", os.path.basename(p))
        if m and (int(m.group(1)) not in best or m.group(2) > best[int(m.group(1))][0]):
            best[int(m.group(1))] = (m.group(2), p)
    return {pid: json.load(open(p, encoding="utf-8")) for pid, (_, p) in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data", type=Path)
    ap.add_argument("--config", default="config.yaml", type=Path)
    ap.add_argument("--through", type=int, default=None,
                    help="last gameweek to replay (default: every one played)")
    ap.add_argument("--no-save", action="store_true",
                    help="score without writing forecasts into the ledger")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace ledger entries that already exist. A live "
                         "pre-deadline forecast is uncontaminated evidence and "
                         "a replay is not, so this is off by default.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.horizon_gw = 1
    cfg.rank_sims = 0          # the replay measures the projection, not the rank layer
    cache = args.root / "cache"

    bootstrap = newest_cached(str(cache / "bootstrap-static_*.json"))
    raw_fixtures = newest_cached(str(cache / "fixtures_*.json"))
    summaries = load_summaries(cache)

    players = normalize_players(bootstrap)
    teams = normalize_teams(bootstrap)
    past = history_past_frame(summaries)
    players = apply_season_baseline(players, past, latest_season(past))
    fixtures = normalize_fixtures(raw_fixtures)

    actuals = actuals_frame(summaries)
    played = sorted(int(r) for r in actuals["round"].unique() if r > 0)
    if args.through:
        played = [gw for gw in played if gw <= args.through]
    if not played:
        raise SystemExit("no played gameweeks found in the cached summaries")

    averages = {int(e["id"]): float(e.get("average_entry_score") or 0)
                for e in bootstrap.get("events", [])}

    writable = set(replayable_gameweeks(played, args.root, overwrite=args.overwrite))
    protected = [gw for gw in played if gw not in writable]
    print(f"Replaying GW{played[0]}..GW{played[-1]} "
          f"({len(players)} players, {len(summaries)} summaries)")
    if protected and not args.no_save:
        kept = ", GW".join(str(g) for g in protected)
        print(f"Keeping the live forecast already on file for GW{kept} "
              f"(scored below, not rewritten; --overwrite to replace).")
    print()
    print(f"{'GW':>3}{'squad':>8}{'field':>7}{'edge':>7}  calibration")
    results = {}
    for gw in played:
        seen = forecast_inputs(summaries, before_event=gw)
        ratings = team_ratings(players, teams, current=seen["current"])
        tfx = team_fixture_frame(fixtures, ratings, gw, 1,
                                 league_gc=league_goals_per_team_match(players))
        rates = blended_rates(players, seen["current"], cfg, rounds=seen["rounds"])
        mins = minutes_model(players, cfg, current=seen["current"], rounds=seen["rounds"])
        xp = build_xp(players, rates, mins, tfx, cfg, gw)

        # Calibrate on gameweeks strictly before this one, exactly as a live run
        # would -- fitting on the gameweek being predicted would be circular.
        note = "-"
        if cfg.calibrate and not args.no_save:
            cal = fit_calibration(scored_history(args.root, summaries, gw))
            if cal is not None:
                xp = apply_calibration(xp, cal)
                note = f"fitted on {cal.n_gameweeks} GW"
        if not args.no_save and gw in writable:
            # Stamped as a replay so the ledger never loses track of which
            # forecasts were made before the deadline and which were
            # reconstructed afterwards from data the live model never had.
            save_predictions(xp, gw, args.root, cfg=cfg,
                             sources={"origin": "replay",
                                      "replayed_at_gw": max(played)})

        frame = xp.merge(actuals[actuals["round"] == gw][["player_id", "actual", "minutes"]],
                         on="player_id", how="inner").set_index("player_id")
        pool = xp[xp["player_id"].isin(frame.index)].copy()
        pool = pool.drop(columns=[c for c in pool.columns if c.startswith("xp_gw")],
                         errors="ignore")
        squad = optimize_squad(pool, cfg, xp_col="xp_next1")
        got = realised_score(squad.player_ids, squad.starting_ids, frame)
        avg = averages.get(gw, 0.0)
        results[gw] = (got["points"], avg)
        print(f"{gw:>3}{got['points']:>8.0f}{avg:>7.0f}{got['points'] - avg:>+7.0f}  {note}")

    ledger = squad_ledger(results)
    verdict = weekly_edge(ledger["edge"].tolist())
    print(f"\nMean edge {verdict['mean']:+.1f} pts/GW over {verdict['n']} gameweeks "
          f"(SD {verdict['sd']:.1f})")
    if verdict["n"] > 1:
        print(f"95% CI [{verdict['ci_low']:+.1f}, {verdict['ci_high']:+.1f}], "
              f"p = {verdict['p']:.3f}")
    print(f"\n{verdict['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
