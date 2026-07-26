#!/usr/bin/env python3
"""Single entry point. Refresh data -> model -> optimize -> report.

Runs headless (cron, `claude -p`) with no MCP and no skills.
"""
import argparse
import sys
from pathlib import Path

from fpl.config import load_config
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

    if args.mode == 2:
        # Mode 2 needs the user's current squad, fetched via
        # entry/{id}/event/{gw}/picks/ -- that fetch isn't wired up in this
        # CLI yet (setting entry_id alone doesn't change that), so this
        # always falls back to a Mode 1 rebuild. pipeline.run() reports the
        # fallback honestly on its own (actual_mode), but tell the user
        # upfront too rather than let them discover it only in the report.
        print(
            "Mode 2 (weekly transfers) isn't wired up in this CLI yet -- it needs "
            "your current squad, which requires fetching entry/{id}/picks/ (not yet "
            "implemented). Falling back to Mode 1 (full squad build) for now.\n"
        )

    rec, xp = run(cfg, mode=args.mode, from_event=args.gw, root=ROOT / "data")
    print(render(rec, xp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
