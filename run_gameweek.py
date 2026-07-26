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
from fpl.data.client import FplClient
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_refresh:
        cfg.cache_ttl_hours = 24 * 365

    data_root = ROOT / "data"
    client = None
    current_squad = None
    bank = 0.0
    free_transfers = cfg.free_transfers

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

    rec, xp = run(cfg, mode=args.mode, from_event=args.gw, root=data_root, client=client,
                  current_squad=current_squad, bank=bank, free_transfers=free_transfers)
    print(render(rec, xp))

    if args.mode == 2 and current_squad is not None:
        chip = rec.chip.chip if rec.chip else None
        transfers_made = rec.transfers.n_transfers if rec.transfers else 0
        record_transfers(data_root / "state.json", cfg, args.gw, transfers_made, chip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
