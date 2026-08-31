#!/usr/bin/env python3
"""Single entry point. Refresh data -> model -> optimize -> report.

Runs headless (cron, `claude -p`) with no MCP and no skills.
"""
import argparse
import sys
from pathlib import Path

from fpl.cli import resolve_current_squad, record_transfers, carry_purchase_prices
from fpl.config import load_config
from fpl.data.cache import Cache
from fpl.data.client import FplClient
from fpl.data.overrides import load_overrides
from fpl.pipeline import run
from fpl.report.weekly import render

ROOT = Path(__file__).parent


def main() -> int:
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
    ap.add_argument("--no-refresh", action="store_true", help="use cached data only")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--overrides", type=Path, default=ROOT / "data" / "overrides.yaml",
                    help="team-news p_start overrides (see fpl/data/overrides.py)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_refresh:
        cfg.cache_ttl_hours = 24 * 365

    data_root = ROOT / "data"
    client = None
    current_squad = None
    bank = 0.0
    free_transfers = cfg.free_transfers
    purchase_prices: dict[int, float] = {}

    if args.mode == 2:
        client = FplClient(Cache(data_root / "cache"), ttl_hours=cfg.cache_ttl_hours)
        live, errors = resolve_current_squad(cfg, args.gw, data_root / "state.json", client)
        if live is None:
            for msg in errors:
                print(f"Note: {msg}")
            print("Falling back to Mode 1 (full squad build) for now.\n")
        else:
            for msg in live.warnings:
                print(f"Note: {msg}")
            current_squad, bank, free_transfers = live.current_squad, live.bank, live.free_transfers
            purchase_prices = live.purchase_prices

    def progress(done: int, total: int) -> None:
        # Player history is fetched one request per second on a cold cache, so a
        # first run takes minutes. Say so rather than looking hung.
        if done == 1 or done == total or done % 100 == 0:
            print(f"Fetching player history... {done}/{total}", flush=True)

    news = load_overrides(args.overrides, args.gw)
    for pid, o in news.items():
        print(f"Override: player {pid} p_start -> {o['p_start_override']} "
              f"(blended at news.weight={cfg.news_weight}) — {o['note']}")

    rec, xp = run(cfg, mode=args.mode, from_event=args.gw, root=data_root, client=client,
                  current_squad=current_squad, bank=bank, free_transfers=free_transfers,
                  news=news, progress=progress, purchase_prices=purchase_prices)
    print(render(rec, xp))

    if args.mode == 2 and current_squad is not None:
        chip = rec.chip.chip if rec.chip else None
        transfers_made = rec.transfers.n_transfers if rec.transfers else 0
        # Carry each retained player's original purchase price forward; a player
        # bought this week was bought at today's price. Without this the selling
        # value resets to market value every run and the budget drifts high again.
        # Record the squad the manager OWNS, not `rec.squad_ids` -- the
        # recommendation is a proposal, and writing it here booked transfers the
        # user never made (see carry_purchase_prices).
        now = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
        updated = carry_purchase_prices(current_squad, purchase_prices, now)
        record_transfers(data_root / "state.json", cfg, args.gw, transfers_made, chip,
                         purchase_prices=updated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
