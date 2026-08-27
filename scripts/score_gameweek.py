#!/usr/bin/env python3
"""Score a recorded forecast against what actually happened.

Reads the prediction ledger written by every pipeline run
(`data/predictions/gw{n}.parquet`), fetches the gameweek's real returns from
element-summary history, and reports rank quality, error and positional bias.

This is the only measurement of the PRODUCTION model. scripts/run_backtest.py
validates a simplified points-per-90 proxy of it, which is why its goalkeeper
verdict (Spearman 0.034) did not survive contact with a real gameweek.

The verdict is saved to `data/predictions/last_score.txt`, which the weekly
report then quotes in place of the hard-coded trust note.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fpl.backtest.ledger import (available_gameweeks, load_predictions,
                                 actuals_from_summaries, score_gameweek,
                                 scored_summary, save_scored_summary)
from fpl.config import load_config
from fpl.data.cache import Cache
from fpl.data.client import FplClient

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Score a recorded gameweek forecast")
    ap.add_argument("--gw", type=int, default=None,
                    help="gameweek to score (default: the latest one in the ledger)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--no-refresh", action="store_true", help="use cached data only")
    ap.add_argument("--quiet", action="store_true",
                    help="save the verdict without printing per-player progress")
    args = ap.parse_args()

    recorded = available_gameweeks(DATA_ROOT)
    if not recorded:
        print("No forecasts recorded yet. Run run_gameweek.py at least once — every "
              "run now writes data/predictions/gw{n}.parquet.")
        return 1

    gw = args.gw if args.gw is not None else recorded[-1]
    if gw not in recorded:
        print(f"No forecast recorded for GW{gw}. Recorded: "
              f"{', '.join(str(g) for g in recorded)}")
        return 1

    cfg = load_config(args.config)
    ttl = 24 * 365 if args.no_refresh else cfg.cache_ttl_hours
    client = FplClient(Cache(DATA_ROOT / "cache"), ttl_hours=ttl)

    pred = load_predictions(gw, DATA_ROOT)
    player_ids = [int(i) for i in pred["player_id"]]

    def progress(done: int, total: int) -> None:
        if not args.quiet and (done == 1 or done == total or done % 100 == 0):
            print(f"Fetching results... {done}/{total}", flush=True)

    summaries = client.element_summaries(player_ids, progress=progress)
    actuals = actuals_from_summaries(summaries, gw)
    if len(actuals) == 0:
        print(f"GW{gw} has no results yet — nothing to score.")
        return 1

    try:
        scored = score_gameweek(pred, actuals)
    except ValueError as e:
        print(str(e))
        return 1

    text = scored_summary(scored, gw)
    print("\n" + text)
    path = save_scored_summary(text, DATA_ROOT)
    print(f"\nSaved to {path}. The next weekly report will quote this instead of the "
          f"hard-coded trust note.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
