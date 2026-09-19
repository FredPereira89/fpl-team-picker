#!/usr/bin/env python
"""Replay past gameweeks and report whether the model actually beats the field.

    python scripts/run_walkforward.py --through 8

Two numbers come out, and they answer different questions.

The first is a SEQUENTIAL replay: one squad carried from gameweek to gameweek
under the real rules -- one bank, recorded purchase prices, a free-transfer
balance, points hits, chip legality, Free Hit restoration. Policies are
compared on the decisions they would actually have made, so the result is
something a manager could have executed.

The second is the free weekly rebuild this script used to report as its
headline. It buys fifteen players from a fresh budget every gameweek and obeys
none of the above, so it is a CEILING and is now labelled as one. The gap
between the two is the cost of having to play by the rules.

Both are still built from today's bootstrap for any gameweek with no
pre-deadline snapshot on file, which means they know about injuries, price
changes and transfers the live model could not. `fpl.data.snapshots` records
snapshots from now on; until enough exist, the contamination banner prints and
any edge reported here is an upper bound.

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
from fpl.model.strength import team_ratings, league_goals_per_team_match
from fpl.model.minutes import minutes_model
from fpl.model.scoring import blended_rates
from fpl.model.fixtures import team_fixture_frame
from fpl.model.xp import build_xp
from fpl.backtest.ledger import save_predictions
from fpl.backtest.walkforward import (forecast_inputs, actuals_frame, realised_score,
                                      squad_ledger, weekly_edge, replayable_gameweeks,
                                      gameweek_inputs, actioned_snapshot,
                                      replay_calibration, config_for_replay)
from fpl.backtest.replay import (ManagerState, compare_policies, hold_policy,
                                 expected_points_policy, multi_period_policy,
                                 oracle_rebuild_policy,
                                 initial_state)
from fpl.data import snapshots
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
                    help="score without writing forecasts into the ledger. Does NOT "
                         "change calibration; the configured challenger setting is "
                         "used unless --no-calibrate is supplied.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="force off per-position calibration (off by default; set "
                         "model.calibrate: true to evaluate the challenger)")
    ap.add_argument("--policies", default="hold,expected,multiperiod",
                    help="comma-separated executable policies to replay in "
                         "sequence (hold, expected, multiperiod). The free weekly rebuild is "
                         "always reported separately as an oracle ceiling.")
    ap.add_argument("--squad", default=None,
                    help="comma/space separated starting 15 for the sequential "
                         "replay (default: the best legal squad at the first "
                         "replayed gameweek, which is itself a small advantage)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace ledger entries that already exist. A live "
                         "pre-deadline forecast is uncontaminated evidence and "
                         "a replay is not, so this is off by default.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # The CONFIGURED horizon, not 1. The production transfer policy chooses on
    # the discounted multi-gameweek xp_horizon; forcing one week here turned
    # the "expected" replay into a greedy weekly chooser that the live tool
    # never runs, and its result then said nothing about the live optimizer.
    # rank_sims stays at 0: since B6 the rank layer only reports on transfers,
    # so switching it off changes no decision this replay makes.
    cfg.rank_sims = 0
    cache = args.root / "cache"

    bootstrap = newest_cached(str(cache / "bootstrap-static_*.json"))
    raw_fixtures = newest_cached(str(cache / "fixtures_*.json"))
    summaries = load_summaries(cache)
    n_players = len(bootstrap.get("elements", []))

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
          f"({n_players} players, {len(summaries)} summaries, "
          f"horizon {cfg.horizon_gw} GW)")
    if protected and not args.no_save:
        kept = ", GW".join(str(g) for g in protected)
        print(f"Keeping the live forecast already on file for GW{kept} "
              f"(scored below, not rewritten; --overwrite to replace).")
    print()
    # The capture each gameweek's ACTIONED forecast read, where one is on
    # record. Without the pin the replay reads the newest pre-deadline capture,
    # which is not necessarily the one the decision was made on.
    pinned = {gw: actioned_snapshot(args.root, gw) for gw in played}
    banner = snapshots.contamination_note(args.root, played, versions_by_gw=pinned)
    if banner:
        print(banner)
        print()
    print(snapshots.SUMMARIES_NOTE)
    print()

    print(f"{'GW':>3}{'squad':>8}{'field':>7}{'edge':>7}  inputs          calibration")
    results = {}
    xp_by_gw = {}
    cfg_by_gw = {}
    archived_by_gw = {}
    config_notes = []
    for gw in played:
        # Prices, availability, news, clubs and fixtures AS OF THE DEADLINE when
        # a pre-deadline snapshot exists; today's otherwise, and flagged. The
        # earlier script imported the snapshot module and then read only the
        # current cache, so a valid snapshot silenced the banner without
        # changing a single input.
        inputs = gameweek_inputs(args.root, gw, bootstrap, raw_fixtures, summaries,
                                 snapshot_version=pinned.get(gw))
        players, teams, fixtures = inputs["players"], inputs["teams"], inputs["fixtures"]
        # The configuration this gameweek was DECIDED under, from its snapshot.
        # Today's settings for a gameweek with none -- which is then a
        # current-configuration challenger, not an exact replay, and says so.
        week_cfg, changed = config_for_replay(cfg, inputs["config"])
        # The rank layer is not run in the replay. Since B6 it only REPORTS on
        # transfers, so switching it off changes no decision -- unless the
        # archived config had opted back into rank-decided transfers, which
        # this replay does not implement. That week is then a current-policy
        # challenger, not a replay of what the tool did, and is labelled.
        archived = bool(inputs["config"])
        if bool(getattr(week_cfg, "rank_transfers", False)):
            origin = "its archived config" if archived else "today's fallback config"
            config_notes.append(f"GW{gw}: {origin} has rank_transfers=true; the replay "
                                f"does not implement rank-decided transfers, so this "
                                f"week runs the expected-points policy (current-policy "
                                f"challenger, not an exact replay)")
        week_cfg.rank_sims = 0
        week_cfg.rank_transfers = False
        cfg_by_gw[gw] = week_cfg
        archived_by_gw[gw] = archived
        if archived and changed:
            config_notes.append(f"GW{gw}: replayed under its archived config "
                                f"({', '.join(changed)} differ from today's)")
        seen = forecast_inputs(summaries, before_event=gw)
        ratings = team_ratings(players, teams, current=seen["current"])
        tfx = team_fixture_frame(fixtures, ratings, gw, week_cfg.horizon_gw,
                                 league_gc=league_goals_per_team_match(players))
        rates = blended_rates(players, seen["current"], week_cfg, rounds=seen["rounds"])
        mins = minutes_model(players, week_cfg, news=inputs["news"],
                             current=seen["current"], rounds=seen["rounds"])
        xp = build_xp(players, rates, mins, tfx, week_cfg, gw)

        # Calibrate on gameweeks strictly before this one, exactly as a live run
        # would -- fitting on the gameweek being predicted would be circular.
        # Independent of --no-save: reading the ledger is not writing it.
        if args.no_calibrate:
            note = "off"
        else:
            xp, note = replay_calibration(xp, args.root, summaries, gw, week_cfg)
        if not args.no_save and gw in writable:
            # Stamped as a replay so the ledger never loses track of which
            # forecasts were made before the deadline and which were
            # reconstructed afterwards from data the live model never had.
            save_predictions(xp, gw, args.root, cfg=week_cfg, origin="replay",
                             sources={"origin": "replay",
                                      "replayed_at_gw": max(played)})

        frame = xp.merge(actuals[actuals["round"] == gw][["player_id", "actual", "minutes"]],
                         on="player_id", how="inner").set_index("player_id")
        # The FULL frame -- horizon columns included -- goes to the sequential
        # replay, so the production policy sees exactly what a live run sees.
        # Each one-week policy strips it back to the current event itself.
        xp_by_gw[gw] = xp[xp["player_id"].isin(frame.index)].copy()
        pool = xp_by_gw[gw].drop(columns=[c for c in xp.columns if c.startswith("xp_gw")],
                                 errors="ignore")
        squad = optimize_squad(pool, week_cfg, xp_col="xp_next1")
        got = realised_score(squad.player_ids, squad.starting_ids, frame)
        avg = averages.get(gw, 0.0)
        results[gw] = (got["points"], avg)
        print(f"{gw:>3}{got['points']:>8.0f}{avg:>7.0f}{got['points'] - avg:>+7.0f}  "
              f"{inputs['source']:<15} {note}")

    # --- what a manager could actually have done -----------------------------
    #
    # This is the comparison that means something. Every policy below starts
    # from one squad and carries its state forward: the same bank, the same
    # purchase prices, the same free-transfer balance, the same chips. The
    # scratch rebuild printed afterwards obeys none of that.
    print("\n" + "=" * 62)
    print("EXECUTABLE POLICIES — one squad, carried forward under the rules")
    print("=" * 62)

    first = played[0]
    # Built and banked under the FIRST week's configuration, not today's: the
    # budget, bench floor and tilt in force at GW1 decide which fifteen the
    # tool would have built, and the whole season descends from them.
    first_cfg = cfg_by_gw[first]
    if args.squad:
        initial = initial_state(xp_by_gw[first], first_cfg,
                                squad=[int(t) for t in args.squad.replace(",", " ").split()])
    else:
        # The production Mode 1 objective -- discounted xp_horizon -- not the
        # one-week column: a squad built on xp_next1 is a different policy from
        # the one the tool would actually have built at GW1.
        initial = initial_state(xp_by_gw[first], first_cfg)
        print(f"No --squad given, so the replay starts from a synthetic Mode 1 "
              f"build at GW{first} (the production squad objective, under GW{first}'s "
              f"configuration). That start is a CHALLENGER, not your season: pass "
              f"--squad with your real GW{first} fifteen for the executable-policy "
              f"claim to be about you.")

    wanted = [n.strip() for n in str(args.policies).split(",") if n.strip()]
    catalogue = {
        "hold": hold_policy,
        "expected": expected_points_policy,
        "multiperiod": multi_period_policy,
    }
    chosen = {n: catalogue[n] for n in wanted if n in catalogue}
    unknown = [n for n in wanted if n not in catalogue]
    if unknown:
        print(f"Ignoring unknown policies: {', '.join(unknown)}")
    chosen["oracle"] = oracle_rebuild_policy

    table = compare_policies(xp_by_gw, actuals, initial, cfg, policies=chosen,
                             field_average=averages, gameweeks=played,
                             cfg_by_gw=cfg_by_gw)
    if config_notes:
        print()
        for line in config_notes:
            print(f"  {line}")
    if any(not archived_by_gw.get(gw) for gw in played):
        print("  Gameweeks without an archived config were replayed under today's "
              "settings: a current-configuration challenger, not an exact replay.")
    print()
    for _, row in table.iterrows():
        tag = "" if row["executable"] else "   <- ORACLE CEILING (not executable)"
        edge = "  n/a" if row["mean_edge"] is None else f"{row['mean_edge']:+6.1f}"
        print(f"  {row['policy']:<10} {row['points']:>7.0f} pts   "
              f"{row['transfers']:>3} transfers  {row['hits_paid']:>3} in hits  "
              f"edge {edge}/GW{tag}")

    executable = table[table["executable"]]
    if len(executable):
        best = executable.iloc[0]
        print(f"\nBest executable policy: {best['policy']} at "
              f"{best['points']:.0f} points.")
    print("The oracle row rebuilds fifteen players from a fresh budget every "
          "gameweek. It is a ceiling, not a strategy, and the gap to it is the "
          "cost of having to obey the rules.")

    # --- the legacy scratch-rebuild number, kept but demoted ------------------
    ledger = squad_ledger(results)
    verdict = weekly_edge(ledger["edge"].tolist())
    print("\n" + "-" * 62)
    print("ORACLE CEILING, scored week by week (the number this script used to "
          "report as its headline)")
    print("-" * 62)
    print(f"Mean edge {verdict['mean']:+.1f} pts/GW over {verdict['n']} gameweeks "
          f"(SD {verdict['sd']:.1f})")
    if verdict["n"] > 1:
        print(f"95% CI [{verdict['ci_low']:+.1f}, {verdict['ci_high']:+.1f}], "
              f"p = {verdict['p']:.3f}")
    print(f"\n{verdict['verdict']}")
    if banner:
        print(f"\n{banner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
