#!/usr/bin/env python3
"""Single entry point. Refresh data -> model -> optimize -> report.

Runs headless (cron, `claude -p`) with no MCP and no skills.
"""
import argparse
import sys
from pathlib import Path

from fpl.cli import resolve_current_squad, record_transfers
from fpl.config import load_config
from fpl.data.cache import Cache
from fpl.data.client import FplClient, FreshDataError
from fpl.data.overrides import load_overrides
from fpl.optimize.transfers import bank_after
from fpl.pipeline import run
from fpl.report.weekly import render
from fpl.backtest import manifest
from fpl.backtest.manifest import mark_actioned
from fpl.chips import CHIPS, chip_blocked_reason
from fpl.state import canonical_chip, load_state

ROOT = Path(__file__).parent
SQUAD_SIZE = 15


def parse_squad(text: str) -> list[int]:
    """Player IDs from a comma- or space-separated list."""
    ids = [int(tok) for tok in text.replace(",", " ").split()]
    if len(set(ids)) != SQUAD_SIZE:
        raise argparse.ArgumentTypeError(
            f"--applied-squad needs exactly {SQUAD_SIZE} distinct player ids, got {len(set(ids))}"
        )
    return ids


def main(argv: list[str] | None = None) -> int:
    # Player names and report formatting can include non-ASCII characters
    # (accents, dashes, currency symbols); Windows consoles often default to
    # a narrow codepage (e.g. cp1252) that can't encode them, crashing the
    # print. Force UTF-8 on stdout so the report always renders.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="FPL gameweek recommendation")
    ap.add_argument("--mode", type=int, choices=(1, 2), default=1,
                    help="1 = full squad build, 2 = weekly transfers")
    ap.add_argument("--gw", type=int, default=1, help="gameweek to optimise for")
    ap.add_argument("--no-refresh", action="store_true",
                    help="allow cached API data and stale fallback (not for live decisions)")
    ap.add_argument("--confirm", action="store_true",
                    help="record this week as PLAYED: store the squad you applied, "
                         "spend the transfers and any chip in data/state.json. "
                         "Without it the run only plans.")
    ap.add_argument("--applied-squad", type=parse_squad, default=None,
                    help="with --confirm: the 15 player ids you ACTUALLY applied, if "
                         "they differ from the recommendation. Transfers and bank are "
                         "derived from this squad rather than assumed.")
    ap.add_argument("--applied-chip", default=None,
                    choices=list(CHIPS) + ["none"],
                    help="with --confirm: the chip you actually played "
                         "(wildcard/freehit/benchboost/triplecaptain), or 'none' if "
                         "you played none. REQUIRED whenever the run advised a "
                         "chip — a recommendation is not an action, and "
                         "confirming one you did not play spends it for "
                         "the season.")
    ap.add_argument("--forecast-version", default=None,
                    help="with --confirm: the forecast version id (from "
                         "data/predictions/manifest.jsonl) the decision was based "
                         "on. Defaults to the newest live forecast made before the "
                         "deadline. A post-deadline version is refused.")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--overrides", type=Path, default=ROOT / "data" / "overrides.yaml",
                    help="team-news p_start overrides (see fpl/data/overrides.py)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.no_refresh:
        # Both, or the matchday TTL the pipeline applies would shorten the very
        # cache --no-refresh exists to keep serving, and the run would go to the
        # network anyway.
        cfg.cache_ttl_hours = cfg.cache_ttl_matchday_hours = 24 * 365

    data_root = ROOT / "data"
    state_path = data_root / "state.json"
    client = FplClient(Cache(data_root / "cache"), ttl_hours=cfg.cache_ttl_hours,
                       fetch_workers=cfg.fetch_workers,
                       fetch_rate_per_s=cfg.fetch_rate_per_s,
                       require_fresh=not args.no_refresh)
    current_squad = None
    bank = 0.0
    free_transfers = cfg.free_transfers
    purchase_prices: dict[int, float] = {}
    # Chips spent so far, DATED. Mode 1 has no live squad to resolve, but a
    # chip is gone whichever mode notices, so local state is read either way.
    chip_events: list[dict] = load_state(state_path, cfg).chip_events
    first_event = 1

    if args.mode == 2:
        live, errors = resolve_current_squad(cfg, args.gw, state_path, client)
        if live is None:
            for msg in errors:
                print(f"Note: {msg}")
            print("Falling back to Mode 1 (full squad build) for now.\n")
        else:
            for msg in live.warnings:
                print(f"Note: {msg}")
            current_squad, bank, free_transfers = live.current_squad, live.bank, live.free_transfers
            purchase_prices = live.purchase_prices
            chip_events, first_event = live.chip_events, live.first_event

    def progress(done: int, total: int) -> None:
        # Player history is fetched a few requests at a time on a cold cache,
        # so a first run still takes a couple of minutes.
        if done == 1 or done == total or done % 100 == 0:
            print(f"Fetching player history... {done}/{total}", flush=True)

    news = load_overrides(args.overrides, args.gw, max_age_hours=cfg.news_max_age_hours)
    for pid, o in news.items():
        if o.get("stale"):
            print(f"WARNING: team news for player {pid} was last checked "
                  f"{o.get('checked_at') or 'never'}, beyond news.max_age_hours="
                  f"{cfg.news_max_age_hours} — re-verify it or drop it. It is "
                  f"still being applied.")
        print(f"Override: player {pid} p_start -> {o['p_start_override']} "
              f"(blended at news.weight={cfg.news_weight}) — {o['note']}")

    # The forecast the manager DECIDED on is the one on file BEFORE this run
    # writes its own. A confirmation re-runs the whole pipeline, and that new
    # forecast would otherwise be the newest pre-deadline version and get
    # marked in place of the planning forecast actually looked at. Captured
    # here, before anything is written, so it cannot be displaced.
    planned = (manifest.select_version(data_root, args.gw)
               if args.confirm and args.forecast_version is None else None)

    try:
        rec, xp = run(cfg, mode=args.mode, from_event=args.gw, root=data_root, client=client,
                      current_squad=current_squad, bank=bank, free_transfers=free_transfers,
                      news=news, progress=progress, purchase_prices=purchase_prices,
                      chip_events=chip_events, first_event=first_event)
    except FreshDataError as exc:
        print(f"Fresh FPL refresh failed: {exc}", file=sys.stderr)
        return 1
    print(render(rec, xp))
    written_versions = manifest.entries(data_root, args.gw)
    if written_versions:
        # Ready to copy into --forecast-version on confirmation.
        print(f"\nForecast version: {written_versions[-1]['version']}")

    # A run is a PROPOSAL, not an execution. Recording unconditionally spent
    # transfers that were only suggested and, when the advisor named a chip,
    # marked that chip used -- a Triple Captain can be lost to a run that was
    # never acted on. Only --confirm writes.
    if args.mode == 2 and current_squad is not None and not args.confirm:
        print("\n(Planning run — data/state.json untouched. Re-run with --confirm "
              "once you have actually made these moves in FPL.)")
    elif args.mode == 2 and current_squad is not None:
        # What was actually applied, which is not necessarily what was advised.
        # Assuming every recommendation was taken is how state drifts away from
        # the real team; --applied-squad/--applied-chip say otherwise.
        applied = list(args.applied_squad or rec.squad_ids)
        # A run is a PROPOSAL. Defaulting to the advised chip spent a Triple
        # Captain on a run that was never acted on -- and for a Free Hit it also
        # decided whether the permanent squad survived. So when the run advised
        # a chip, the manager has to say what they actually played.
        advised = rec.chip.chip if rec.chip else None
        if args.applied_chip is None:
            if advised is not None:
                print(f"\nThis run advised {advised}. Re-run with "
                      f"--applied-chip {advised} if you played it, or "
                      f"--applied-chip none if you did not. Nothing was recorded.")
                return 1
            chip = None
        else:
            chip = None if str(args.applied_chip).lower() == "none" else \
                canonical_chip(args.applied_chip)
        # The advisor and the replay both refuse an illegal chip; confirmation
        # must too, or a typo or a second same-window chip lands in state and
        # every later run reasons from a chip history that never happened.
        # A same-gameweek re-confirmation of the SAME chip is idempotent, not a
        # second use, so that one record is set aside. Only that one: dropping
        # every same-gameweek record let a recorded Bench Boost be replaced by a
        # Wildcard in the same week, which breaks one-active-chip.
        earlier = [e for e in chip_events
                   if not (e.get("event") == args.gw
                           and canonical_chip(e.get("chip")) == chip)]
        blocked = chip_blocked_reason(chip, args.gw, earlier, first_event) if chip else None
        if blocked:
            print(f"\nRefusing to record {chip} for GW{args.gw}: {blocked}. "
                  f"Nothing was written.")
            return 1
        transfers_made = len(set(current_squad) - set(applied))
        now = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
        cash = (rec.bank if applied == list(rec.squad_ids)
                else bank_after(bank, current_squad, applied, now, purchase_prices))
        # Carry each retained player's original purchase price forward; a player
        # bought this week was bought at today's price. Without this the selling
        # value resets to market value every run and the budget drifts high again.
        updated = {int(pid): float(purchase_prices.get(int(pid), now[int(pid)]))
                   for pid in applied}
        # The pre-deadline squad, bank and prices are what FPL restores after a
        # Free Hit, so they are handed over as the permanent state to keep.
        written = record_transfers(state_path, cfg, args.gw, transfers_made, chip,
                                   purchase_prices=updated, squad=applied, bank=cash,
                                   api_chips=chip_events,
                                   base_squad=list(current_squad),
                                   base_bank=bank,
                                   base_purchase_prices=dict(purchase_prices))
        # Name the forecast that was acted on. This run has just written its
        # OWN forecast, so "newest" would be the confirmation's, not the planning
        # one the manager looked at -- hence the deadline: the default is the
        # newest live version made before it, and a post-deadline version is
        # refused so it cannot carry the team news into calibration.
        deadline = rec.deadline if "T" in str(rec.deadline) else None
        version = args.forecast_version or (planned["version"] if planned else None)
        marked = mark_actioned(data_root, args.gw, version=version, deadline=deadline)
        if marked is None and version is not None and args.forecast_version is None:
            # The planning forecast was after the deadline; fall back to the
            # rules mark_actioned applies on its own, which will refuse too if
            # nothing pre-deadline exists.
            marked = mark_actioned(data_root, args.gw, deadline=deadline)
        if marked is None:
            print("\nWARNING: no pre-deadline forecast could be marked as the one "
                  "acted on (this run is after the deadline, or the named "
                  "--forecast-version was not eligible). The squad and chip were "
                  "recorded; the ledger will score the newest pre-deadline "
                  "forecast on file, if any.")
        print(f"\nRecorded GW{args.gw} as played: {transfers_made} transfer(s), "
              f"chip {chip or 'none'}, bank £{written.bank}m, "
              f"{written.free_transfers_remaining} free transfer(s) left this week, "
              f"{written.free_transfers} for GW{args.gw + 1}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
