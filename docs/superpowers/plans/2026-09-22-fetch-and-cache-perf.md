# Fetch and Cache Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the weekly FPL run's refresh from 11+ minutes to about 2.5 and its cache-only run from 40.6 s to ≤ 20 s, with no change to any recommendation.

**Architecture:** First measure (a bench harness, a clock-frozen GW6 golden output, and complexity tests that fail on today's code). Then replace the per-lookup directory `glob` in `Cache` with a slug→paths index. It is validated by the directory's mtime and heals itself if a snapshot vanishes. Then fetch missing element-summaries through a bounded, cancellable thread pool behind a shared token-bucket limiter with 429/5xx backoff that honours `Retry-After`. HTTP runs on worker threads; every cache write and piece of client bookkeeping stays on the main thread. An optional last phase vectorises the xP threshold tail.

**Tech Stack:** Python 3.13, pandas 2.2, numpy 1.26, scipy 1.14, requests 2.31, pytest 8, and `concurrent.futures` / `threading` / `email.utils` from the stdlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`

**Revision 2 (2026-09-22)** addresses the plan review of `f7c6a93`:
1. The limiter no longer reserves slots, so a pause holds back threads already waiting.
2. An interrupt now stops workers (cancel event, bounded submission, sessions closed only after workers exit).
3. The behaviour contract is restated honestly (retries and de-duplication are intentional changes).
4. The cache index notices other writers through the directory mtime and heals a vanished snapshot.
5. `Retry-After` accepts HTTP-dates and is never shortened.
6. The golden run freezes the clock.
7. The refresh bench uses `config.yaml`'s fetch settings.

## Global Constraints

- Work only in the worktree `C:\Users\user\Documents\FPL Team Picker\.claude\worktrees\perf-fetch-and-cache` (branch `worktree-perf-fetch-and-cache`). Never run git against the main checkout.
- No new third-party dependencies (`requirements.txt` stays unchanged).
- **Behaviour contract.** Whenever every fetch succeeds, or fails permanently (4xx other than 429, retries exhausted, or no network), `element_summaries` leaves the same dict, `fetch_failures`, `stale`, `unverified` and `sources` as today's code, including per-player containment: any cache read, write, prune or fallback error fails only that player. There are exactly **three intentional differences**:
  - Transient failures (429, 5xx, connection errors, timeouts) are retried up to 3 times before counting as failures.
  - Duplicate ids are fetched once, and `progress` totals count unique ids.
  - A `Retry-After` longer than 120 s stops the element-summary fetch for the rest of the run. The players not yet fetched take the stale-fallback / `fetch_failures` path instead of being attempted (today every later player would still be tried).

  The GW6 golden decision must stay identical, and every numeric xP column must match within **1e-4 absolute**, with the clock frozen at capture time.
- All existing tests pass after every task (774 at the start; `python -m pytest -q -p no:cacheprovider`).
- Defaults: `fetch_workers: 4`, `fetch_rate_per_s: 5.0`. `fetch_workers: 1` with `fetch_rate_per_s: 1.0` restores today's **request pacing**: one request in flight, starts at least 1 s apart. Retries and de-duplication remain.
- Only `FplClient.element_summaries` becomes concurrent. `_get`, `bootstrap`, `fixtures`, `entry*` and the `rate_limit_s` throttle keep their current behaviour.
- All cache and client-state mutation happens on the main thread. Worker threads only perform HTTP.
- A server's `Retry-After` is never shortened. A wait of ≤ 120 s (`MAX_RETRY_AFTER_S`) is sat out. A longer one ends the element-summary fetch for this run, and the remaining players take the stale-fallback / `fetch_failures` path, the same as in an outage.
- `Cache` assumes one writer per directory at a time. Other writers are detected best-effort through the directory mtime (NTFS timestamps are tick-granular). A snapshot deleted by someone else causes one rescan, never a crash.
- Tests assert complexity (scan counts, in-flight counts, rate windows), never absolute wall time on real data.
- The code style matches the repo: comments explain *why*, in full sentences, at the density of the surrounding code.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Ctrl-C (or any exception) in the middle of a 667-player refresh**, including while a worker sits in a retry backoff, must stop the *workers*, not just return from the call. No request may start after the call has raised; already-fetched summaries must be on disk. → Task 6, `test_an_interrupted_refresh_stops_its_workers` and `test_an_interrupt_cuts_a_retry_backoff_short`.
2. **FPL answers a burst with 429 while other workers are already waiting for their slot.** The pause must hold them too, and the throttled request is retried, not failed. → Task 5, `test_a_pause_holds_back_threads_already_waiting`; Task 6, `test_429_pauses_the_limiter_and_retries`.
3. **`Retry-After` as an HTTP-date, or longer than a run can wait.** The date is parsed, and a long wait is obeyed by not asking again, with the remaining players falling back to their cached snapshots. → Task 6, `test_retry_after_accepts_seconds_and_http_dates` and `test_a_long_retry_after_stops_asking_and_falls_back`.
4. **Players whose request keeps failing** (404 for a removed player, persistent 503) end in the stale fallback or in `fetch_failures` after a bounded number of tries; a 404 is never retried. → Task 6, `test_404_is_not_retried` and `test_persistent_503_gives_up_after_bounded_retries`.
5. **The cache folder is changed by something other than this `Cache`**: files that aren't snapshots, prefix-sharing slugs, another writer adding a snapshot, or deleting one the index still lists. Each lookup stays correct and never crashes. → Task 4, `test_index_ignores_files_that_are_not_snapshots`, `test_index_keeps_prefix_sharing_slugs_apart`, `test_another_writer_is_noticed` and `test_a_snapshot_deleted_elsewhere_heals_instead_of_crashing`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/bench.py` | create | Benchmarks (`run`) and the GW6 golden capture and check (`golden capture` / `golden check`). All pipeline runs happen in a temporary copy of the repo. |
| `docs/perf/README.md` | create | The before/after table for every phase. |
| `docs/perf/*.json` | create | Raw benchmark output, one file per `bench.py run --label`. |
| `docs/perf/golden-gw6/decision.json`, `xp.parquet`, `clock.txt` | create | The frozen GW6 decision and xP frame from unmodified master, and the instant the clock is frozen at for every check. |
| `tests/test_perf_contracts.py` | create | Complexity contracts: directory scans, in-flight bound, rate cap, real concurrency. |
| `tests/test_client.py` | modify | Parity tests pinning `element_summaries` semantics, plus retry and interrupt tests. |
| `tests/test_coverage_gate.py` | modify | Replace the implementation-coupled client test with a behavioural one. |
| `tests/test_cache.py` | modify | Index behaviour tests. |
| `tests/test_throttle.py` | create | `TokenBucket` unit tests. |
| `tests/test_config.py` | modify | The new `data.fetch_*` keys. |
| `fpl/data/cache.py` | modify | Slug→paths index replacing the per-lookup `glob`; validated by directory mtime, self-healing on a vanished snapshot. |
| `fpl/data/throttle.py` | create | `TokenBucket` (thread-safe, cancellable, no reservations, so a pause holds waiting threads) and `FetchCancelled`. |
| `fpl/data/client.py` | replace | Concurrent `element_summaries` via the bounded, cancellable `_fetch_misses`; the retrying `_fetch_json` with `Retry-After`; per-thread sessions; shared helpers `_cached` / `_store` / `_fallback`. |
| `fpl/config.py`, `config.yaml` | modify | `fetch_workers`, `fetch_rate_per_s`. |
| `run_gameweek.py`, `fpl/pipeline.py`, `scripts/run_backtest.py`, `scripts/score_gameweek.py` | modify | Pass the fetch settings to `FplClient`. |
| `fpl/model/xp.py` | modify (Task 7, optional) | Vectorised threshold tail; per-team fixture grouping; no per-fixture copy. |

---

### Task 1: Freeze inputs, bench harness, golden capture, baseline numbers

**Files:**
- Create: `scripts/bench.py`
- Create: `docs/perf/README.md`, `docs/perf/baseline.json` (generated), `docs/perf/golden-gw6/decision.json`, `xp.parquet` and `clock.txt` (generated)
- Commit also: `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`, this plan

**Interfaces:**
- Consumes: `fpl.data.cache.Cache`, `fpl.data.client.FplClient`, `run_gameweek.main(argv) -> int`, `run_gameweek.run` (module global, rebound by the golden driver), the `Recommendation` fields `squad_ids`, `lineup.{xi,bench,formation,captain,vice}`, `transfers.{out_ids,in_ids}` and `chip.chip`.
- Produces: `python scripts/bench.py run --label <name>` → `docs/perf/<name>.json` with keys `label, commit, python, machine, when, cache_hits_s, refresh, gw6_cache_only_s`. `python scripts/bench.py golden capture` and `python scripts/bench.py golden check` (exit 0 = parity, 1 = drift, and each problem printed). Later tasks run these commands at their end.

- [ ] **Step 1: Freeze this week's data inside the worktree**

`data/cache` and `data/processed` are gitignored, so the worktree has neither. Copy them in once. From then on the golden run's inputs can't drift when the main checkout refreshes for GW7.

```bash
cp -r "/c/Users/user/Documents/FPL Team Picker/data/cache" data/cache
cp -r "/c/Users/user/Documents/FPL Team Picker/data/processed" data/processed
ls data/cache | wc -l          # expect ~4061
git status --short             # expect nothing new: both are ignored
```

- [ ] **Step 2: Write `scripts/bench.py`**

```python
#!/usr/bin/env python3
"""Performance benchmarks and the GW6 golden-output check.

    python scripts/bench.py run --label baseline   # time everything -> docs/perf/baseline.json
    python scripts/bench.py golden capture         # freeze the GW6 decision + xP frame
    python scripts/bench.py golden check           # diff the current code against it

Every pipeline run happens in a temporary copy of the repo. A plan-only run
still writes a forecast version, and data/predictions must never collect
benchmark entries -- the backtest reads that manifest as the record of what was
forecast.
"""
import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fpl.config import load_config  # noqa: E402
from fpl.data.cache import Cache  # noqa: E402
from fpl.data.client import FplClient  # noqa: E402

PERF = ROOT / "docs" / "perf"
GOLDEN = PERF / "golden-gw6"
GOLDEN_CLOCK = GOLDEN / "clock.txt"
GW = 6
N_PLAYERS = 667
REFRESH_N = 30
REFRESH_LATENCY_S = 0.15
XP_TOL = 1e-4

# Runs inside the temporary repo copy. It wraps `run` so the Recommendation and
# the xP frame can be captured without teaching run_gameweek a new flag, and it
# freezes the clock at the instant recorded when the golden output was
# captured: override ages (fpl/data/overrides.py) and the matchday check
# (fpl/pipeline.py) read datetime.now(), so without this a golden check a few
# days later could drift with no code change at all.
DRIVER = r'''
import importlib, json, pkgutil, sys
from datetime import datetime, timezone

import fpl
for info in pkgutil.walk_packages(fpl.__path__, "fpl."):
    importlib.import_module(info.name)       # patch below reaches every module
import run_gameweek as rg

FROZEN = datetime.fromisoformat(sys.argv[2])


class _FrozenMeta(type):
    # A plain subclass would make isinstance(real_datetime, datetime) False in
    # every patched module -- fpl.data.overrides._as_datetime depends on it.
    def __instancecheck__(cls, obj):
        return isinstance(obj, datetime)


class FrozenDateTime(datetime, metaclass=_FrozenMeta):
    @classmethod
    def now(cls, tz=None):
        return FROZEN.astimezone(tz) if tz is not None else FROZEN.astimezone().replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return FROZEN.astimezone(timezone.utc).replace(tzinfo=None)


for mod in list(sys.modules.values()):
    name = getattr(mod, "__name__", "") or ""
    if (name == "run_gameweek" or name.startswith("fpl.")) and getattr(mod, "datetime", None) is datetime:
        mod.datetime = FrozenDateTime

captured = {}
_orig = rg.run


def _spy(*a, **k):
    out = _orig(*a, **k)
    captured["rec"], captured["xp"] = out
    return out


rg.run = _spy
code = rg.main(["--mode", "2", "--gw", sys.argv[1], "--no-refresh"])
rec, xp = captured["rec"], captured["xp"]
tr = rec.transfers
decision = {
    "exit_code": int(code or 0),
    "squad_ids": sorted(int(i) for i in rec.squad_ids),
    "xi": sorted(int(i) for i in rec.lineup.xi),
    "bench": [int(i) for i in rec.lineup.bench],
    "formation": str(rec.lineup.formation),
    "captain": int(rec.lineup.captain),
    "vice": int(rec.lineup.vice),
    "transfers_out": sorted(int(i) for i in tr.out_ids) if tr else [],
    "transfers_in": sorted(int(i) for i in tr.in_ids) if tr else [],
    "chip": getattr(rec.chip, "chip", None),
}
with open("decision.json", "w") as fh:
    json.dump(decision, fh, indent=2)
xp.to_parquet("xp.parquet")
'''


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class _LatencySession:
    """Answers every request after a fixed delay: a stand-in for FPL's API."""

    def __init__(self, latency_s: float):
        self.latency_s = latency_s

    def get(self, url, timeout=None):
        time.sleep(self.latency_s)
        return _Resp({"history": [], "history_past": []})


class _NoNetwork:
    def get(self, url, timeout=None):
        raise AssertionError(f"benchmark went to the network: {url}")


def synthetic_cache(root: Path, n_players: int = N_PLAYERS, snapshots: int = 3,
                    noise: int = 2000) -> list[int]:
    """A cache shaped like the real one: n players x 3 snapshots with sidecars,
    plus unrelated files so every directory scan pays for a realistic size."""
    cache = Cache(root)
    now = datetime.now(timezone.utc)
    for pid in range(1, n_players + 1):
        for k in range(snapshots):
            cache.put(f"element-summary-{pid}", {"history": [], "history_past": []},
                      now=now - timedelta(hours=k + 1), meta={"final_through": GW - 1})
    for i in range(noise):
        (root / f"archive-noise-{i}_20260101T000000Z.json").write_text("{}")
    return list(range(1, n_players + 1))


def bench_cache_hits() -> float:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ids = synthetic_cache(Path(d))
        client = FplClient(Cache(Path(d)), rate_limit_s=0, session=_NoNetwork())
        t = time.perf_counter()
        out = client.element_summaries(ids, require_final_through=GW - 1)
        elapsed = time.perf_counter() - t
    assert len(out) == len(ids), "cache_hits: some lookups missed"
    return elapsed


def _production_fetch_settings() -> dict:
    """The fetch settings a real run would use, read from config.yaml the way
    run_gameweek does. Before phase 2 there are none: the client's 1 s throttle
    IS the production path, so an empty dict measures exactly that."""
    cfg = load_config(ROOT / "config.yaml")
    if not hasattr(cfg, "fetch_workers"):
        return {}
    return {"fetch_workers": cfg.fetch_workers, "fetch_rate_per_s": cfg.fetch_rate_per_s}


def bench_refresh(n: int = REFRESH_N, latency_s: float = REFRESH_LATENCY_S) -> dict:
    """An empty cache fetched at the production settings from config.yaml, then
    extrapolated to 667 players."""
    settings = _production_fetch_settings()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        client = FplClient(Cache(Path(d)), session=_LatencySession(latency_s), **settings)
        t = time.perf_counter()
        out = client.element_summaries(range(1, n + 1))
        elapsed = time.perf_counter() - t
    assert len(out) == n, "refresh: some fetches failed"
    return {"n": n, "latency_s": latency_s, "settings": settings or "sequential, 1 s throttle",
            "seconds": round(elapsed, 3),
            "req_per_s": round(n / elapsed, 2),
            "extrapolated_667_s": round(elapsed / n * N_PLAYERS, 1)}


def _temp_repo(dest: Path) -> Path:
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(ROOT / "fpl", dest / "fpl", ignore=ignore)
    shutil.copy2(ROOT / "run_gameweek.py", dest / "run_gameweek.py")
    shutil.copy2(ROOT / "config.yaml", dest / "config.yaml")
    shutil.copytree(ROOT / "data", dest / "data", ignore=ignore)
    return dest


def _have_real_cache() -> bool:
    if (ROOT / "data" / "cache").is_dir():
        return True
    print("data/cache is absent (it is gitignored): copy it in first -- see plan Task 1 Step 1.")
    return False


def bench_gw6_cache_only() -> float | None:
    if not _have_real_cache():
        return None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = _temp_repo(Path(d))
        t = time.perf_counter()
        subprocess.run([sys.executable, "run_gameweek.py", "--mode", "2", "--gw", str(GW),
                        "--no-refresh"], cwd=repo, check=True, capture_output=True)
        return time.perf_counter() - t


def _median(fn, repeats: int) -> float:
    return statistics.median(fn() for _ in range(repeats))


def cmd_run(label: str) -> int:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    result = {
        "label": label, "commit": commit, "python": platform.python_version(),
        "machine": platform.platform(), "when": datetime.now(timezone.utc).isoformat(),
        "cache_hits_s": round(_median(bench_cache_hits, 3), 3),
        "refresh": bench_refresh(),
    }
    gw6 = [bench_gw6_cache_only() for _ in range(3)]
    result["gw6_cache_only_s"] = (round(statistics.median(gw6), 2)
                                  if all(v is not None for v in gw6) else None)
    PERF.mkdir(parents=True, exist_ok=True)
    out = PERF / f"{label}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"-> {out.relative_to(ROOT)}")
    return 0


def _golden_run(frozen_now: str) -> tuple[dict, pd.DataFrame]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = _temp_repo(Path(d))
        (repo / "golden_driver.py").write_text(DRIVER)
        proc = subprocess.run([sys.executable, "golden_driver.py", str(GW), frozen_now], cwd=repo,
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
            raise SystemExit("golden run failed")
        decision = json.loads((repo / "decision.json").read_text())
        xp = pd.read_parquet(repo / "xp.parquet")
    return decision, xp


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" in df.columns:
        df = df.set_index("player_id")
    return df.sort_index()


def compare_xp(old: pd.DataFrame, new: pd.DataFrame, tol: float = XP_TOL) -> list[str]:
    old, new = _keyed(old), _keyed(new)
    problems = []
    if list(old.columns) != list(new.columns):
        problems.append(f"columns differ: {list(old.columns)} vs {list(new.columns)}")
    if set(old.index) != set(new.index):
        problems.append(f"player sets differ: {len(old)} vs {len(new)} rows")
        return problems
    new = new.loc[old.index]
    for col in old.columns.intersection(new.columns):
        a, b = old[col], new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if (a.isna() != b.isna()).any():
                problems.append(f"{col}: NaN in different rows")
                continue
            worst = (a.astype(float) - b.astype(float)).abs().max()
            if pd.notna(worst) and worst > tol:
                problems.append(f"{col}: max |diff| {worst:.2e} > {tol:.0e}")
        elif a.astype(str).tolist() != b.astype(str).tolist():
            problems.append(f"{col}: values differ")
    return problems


def cmd_golden(action: str) -> int:
    if not _have_real_cache():
        return 1
    if action == "capture":
        frozen_now = datetime.now(timezone.utc).isoformat()
    else:
        frozen_now = GOLDEN_CLOCK.read_text().strip()
    decision, xp = _golden_run(frozen_now)
    if action == "capture":
        GOLDEN.mkdir(parents=True, exist_ok=True)
        GOLDEN_CLOCK.write_text(frozen_now + "\n")
        (GOLDEN / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        xp.to_parquet(GOLDEN / "xp.parquet")
        print(json.dumps(decision, indent=2))
        print(f"-> {GOLDEN.relative_to(ROOT)}")
        return 0
    old = json.loads((GOLDEN / "decision.json").read_text())
    problems = [f"decision.{k}: {old[k]!r} -> {decision.get(k)!r}"
                for k in old if old[k] != decision.get(k)]
    problems += compare_xp(pd.read_parquet(GOLDEN / "xp.parquet"), xp)
    for p in problems:
        print("DRIFT", p)
    print("golden: OK" if not problems else f"golden: {len(problems)} problem(s)")
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    g = sub.add_parser("golden")
    g.add_argument("action", choices=("capture", "check"))
    args = ap.parse_args(argv)
    return cmd_run(args.label) if args.cmd == "run" else cmd_golden(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Capture the golden output and prove it's deterministic**

```bash
python scripts/bench.py golden capture
python scripts/bench.py golden check
```
Expected: `capture` prints the decision (GW6 squad after the Wildcard, captain, and so on) and writes the frozen instant to `clock.txt`. `check` replays the run at that same instant and prints `golden: OK`, exiting 0. (This round-trip, and a planted 1% xP change being flagged as 8 problems, were verified on a scratch copy while revising the plan.) If `check` reports drift on unchanged code, **stop**: the run isn't deterministic (look for simulation seeds or time-dependent inputs), and no later parity claim means anything until that's fixed.

- [ ] **Step 4: Record the baseline numbers**

```bash
python scripts/bench.py run --label baseline
```
Expected: this takes about 4 minutes, most of it in `refresh`, which runs 30 requests under today's 1-per-second throttle. Rough expectations: `cache_hits_s` ≈ 10–25 s, `refresh.extrapolated_667_s` ≈ 650–680, `gw6_cache_only_s` ≈ 38–45.

- [ ] **Step 5: Write `docs/perf/README.md`**

Fill in the baseline column from `docs/perf/baseline.json`. Leave the later columns as `—` until those phases land.

```markdown
# Performance log — fetch and cache

Spec: `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`
Plan: `docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md`

Produced by `python scripts/bench.py run --label <phase>`; the raw JSON sits
next to this file. The golden check is `python scripts/bench.py golden check`.

| Benchmark | baseline | phase1-cache-index | phase2-concurrent-fetch | phase3-xp (optional) |
|---|---|---|---|---|
| cache_hits (667 lookups, synthetic cache), s | <from baseline.json> | — | — | — |
| refresh, extrapolated to 667 players, s | <from baseline.json> | — | — | — |
| refresh, requests/s | <from baseline.json> | — | — | — |
| GW6 Mode-2 `--no-refresh`, s | <from baseline.json> | — | — | — |
| Golden check | OK (captured) | — | — | — |
```
(Replace each `<from baseline.json>` with the number. The README must contain no angle-bracket placeholders when committed.)

- [ ] **Step 6: Commit**

```bash
git add scripts/bench.py docs/perf docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md
git commit -m "perf: bench harness, GW6 golden output and baseline numbers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Complexity-contract tests (xfail-strict against today's code)

**Files:**
- Create: `tests/test_perf_contracts.py`

**Interfaces:**
- Consumes: `Cache(root)`, `FplClient(cache, rate_limit_s=..., session=...)`, and, from Task 6, the constructor keywords `fetch_workers: int` and `fetch_rate_per_s: float | None`.
- Produces: four tests. Task 4 removes the xfail marker from `test_cache_scans_directory_once`; Task 6 removes the other three.

- [ ] **Step 1: Write the tests**

```python
"""Complexity contracts for the cache and the element-summary fetch.

These pin HOW the work scales -- directory scans per lookup, requests in
flight, requests per second -- rather than wall time, so they are
deterministic on any machine. Each starts life as xfail(strict=True) against
the code it is meant to change; the phase that satisfies it removes the marker.
"""
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fpl.data.cache import Cache
from fpl.data.client import FplClient

N = 200


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class RecordingSession:
    """Thread-safe fake that records concurrency and request start times."""

    def __init__(self, latency_s: float = 0.05):
        self.latency_s = latency_s
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.starts: list[float] = []
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.starts.append(time.monotonic())
            self.calls.append(url)
        try:
            time.sleep(self.latency_s)
            return _Resp({"history": [], "url": url})
        finally:
            with self.lock:
                self.in_flight -= 1


class NoNetwork:
    def get(self, url, timeout=None):
        raise AssertionError(f"went to the network: {url}")


def _seed(root: Path, n: int = N) -> list[int]:
    cache = Cache(root)
    now = datetime.now(timezone.utc)
    for pid in range(1, n + 1):
        for k in range(3):
            cache.put(f"element-summary-{pid}", {"history": []},
                      now=now - timedelta(hours=k + 1), meta={"final_through": 5})
    return list(range(1, n + 1))


def _count_scans(monkeypatch, root: Path) -> dict:
    """Count directory listings of `root`, however they are made."""
    calls = {"n": 0}
    real_glob, real_scandir = Path.glob, os.scandir

    def glob(self, pattern, *a, **k):
        if Path(self) == root:
            calls["n"] += 1
        return real_glob(self, pattern, *a, **k)

    def scandir(path=".", *a, **k):
        if Path(path) == root:
            calls["n"] += 1
        return real_scandir(path, *a, **k)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(os, "scandir", scandir)
    return calls


@pytest.mark.xfail(strict=True, reason="phase 1: cache index not built yet")
def test_cache_scans_directory_once(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=NoNetwork())
    scans = _count_scans(monkeypatch, tmp_path)

    out = client.element_summaries(ids, require_final_through=5)

    assert len(out) == N
    # Globbing per lookup listed the whole directory ~3 times per player.
    assert scans["n"] <= 1


@pytest.mark.xfail(strict=True, reason="phase 2: concurrent fetch not built yet")
def test_refresh_is_concurrent(tmp_path):
    s = RecordingSession(latency_s=0.15)
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s,
                  fetch_workers=4, fetch_rate_per_s=0)

    t = time.monotonic()
    out = c.element_summaries(range(1, 41))
    elapsed = time.monotonic() - t

    assert len(out) == 40
    # Sequential is 40 x 0.15 = 6 s; four workers need ~1.5 s.
    assert elapsed < 3.0


@pytest.mark.xfail(strict=True, reason="phase 2: concurrent fetch not built yet")
def test_refresh_respects_worker_bound(tmp_path):
    s = RecordingSession(latency_s=0.05)
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s,
                  fetch_workers=3, fetch_rate_per_s=0)

    c.element_summaries(range(1, 31))

    assert len(s.calls) == 30
    assert 2 <= s.peak <= 3


@pytest.mark.xfail(strict=True, reason="phase 2: concurrent fetch not built yet")
def test_refresh_respects_rate_cap(tmp_path):
    s = RecordingSession(latency_s=0.01)
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s,
                  fetch_workers=8, fetch_rate_per_s=10)

    c.element_summaries(range(1, 26))

    starts = sorted(s.starts)
    assert len(starts) == 25
    # No one-second window may hold more than the cap (plus one for float slop).
    for i, t0 in enumerate(starts):
        in_window = sum(1 for t in starts[i:] if t < t0 + 1.0)
        assert in_window <= 11, f"{in_window} requests started within 1 s of #{i}"
```

- [ ] **Step 2: Run them and confirm all four xfail**

Run: `python -m pytest tests/test_perf_contracts.py -v -p no:cacheprovider`
Expected: `4 xfailed`. The first fails on the scan count; the other three raise `TypeError: unexpected keyword argument 'fetch_workers'`. If any shows **XPASS**, the contract isn't testing what it claims: fix the test before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_perf_contracts.py
git commit -m "test: complexity contracts for cache scans and concurrent fetch (xfail)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pin today's `element_summaries` semantics (parity tests)

**Files:**
- Modify: `tests/test_client.py` (append)
- Modify: `tests/test_coverage_gate.py:69-84`

**Interfaces:**
- Consumes: `FplClient.element_summaries(player_ids, ttl_hours=..., progress=..., not_before=..., require_final_through=...) -> dict[int, dict]` and the attributes `stale: bool`, `fetch_failures: set[int]`, `unverified: set[str]`, `sources: dict[str, str]`, `snapshot_meta: dict`.
- Produces: tests that must pass **before and after** Task 6, unchanged.

- [ ] **Step 1: Append the parity tests to `tests/test_client.py`**

```python
# --- element_summaries semantics --------------------------------------------
# Pinned against the sequential implementation before the fetch went
# concurrent. They must pass, unchanged, on both.

class PerUrlSession:
    """Routes by URL; any URL in `failing` raises like a dropped connection."""

    def __init__(self, routes, failing=()):
        self.routes, self.failing, self.calls = routes, set(failing), []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url in self.failing:
            raise RuntimeError("connection reset")
        return FakeResponse(self.routes[url])


def _es(pid):
    return BASE + f"element-summary/{pid}/"


def test_element_summaries_mixes_hits_and_fetches(tmp_path):
    cache = Cache(tmp_path)
    for pid in (1, 2):
        cache.put(f"element-summary-{pid}", {"id": pid, "src": "cache"},
                  now=now() - timedelta(hours=1))
    s = PerUrlSession({_es(3): {"id": 3, "src": "net"}, _es(4): {"id": 4, "src": "net"}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2, 3, 4])

    assert list(out) == [1, 2, 3, 4]
    assert [out[p]["src"] for p in (1, 2, 3, 4)] == ["cache", "cache", "net", "net"]
    assert sorted(s.calls) == [_es(3), _es(4)]
    assert c.stale is False and c.fetch_failures == set()
    assert set(c.sources) == {f"element-summary-{p}" for p in (1, 2, 3, 4)}
    assert cache.newest("element-summary-3")[0] == {"id": 3, "src": "net"}


def test_element_summaries_writes_the_snapshot_marker(tmp_path):
    s = PerUrlSession({_es(7): {"id": 7}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    c.snapshot_meta = {"final_through": 5}

    c.element_summaries([7])

    assert c.cache.newest_meta("element-summary-7") == {"final_through": 5}


def test_a_failed_fetch_falls_back_to_the_old_snapshot(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-5", {"id": 5, "src": "old"}, now=now() - timedelta(days=11))
    s = PerUrlSession({}, failing=[_es(5)])
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([5], not_before=now() - timedelta(days=2))

    assert out == {5: {"id": 5, "src": "old"}}
    assert c.stale is True
    assert c.fetch_failures == set()
    assert "element-summary-5" in c.sources


def test_a_failed_fetch_with_no_snapshot_is_recorded_and_omitted(tmp_path):
    s = PerUrlSession({_es(1): {"id": 1}}, failing=[_es(6)])
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    out = c.element_summaries([1, 6])

    assert list(out) == [1]
    assert c.stale is True
    assert c.fetch_failures == {6}


def test_a_snapshot_without_the_marker_is_flagged_unverified(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-1", {"id": 1}, now=now() - timedelta(hours=1))
    cache.put("element-summary-2", {"id": 2}, now=now() - timedelta(hours=1),
              meta={"final_through": 5})
    c = FplClient(cache, rate_limit_s=0, session=PerUrlSession({}))

    c.element_summaries([1, 2], require_final_through=5)

    assert c.unverified == {"element-summary-1"}


def test_progress_counts_every_player_once_in_order(tmp_path):
    cache = Cache(tmp_path)
    cache.put("element-summary-1", {"id": 1}, now=now() - timedelta(hours=1))
    s = PerUrlSession({_es(p): {"id": p} for p in (2, 3, 4)}, failing=[_es(5)])
    c = FplClient(cache, rate_limit_s=0, session=s)
    seen = []

    c.element_summaries([1, 2, 3, 4, 5], progress=lambda d, t: seen.append((d, t)))

    assert [d for d, _ in seen] == [1, 2, 3, 4, 5]
    assert {t for _, t in seen} == {5}


# Per-player containment: the sequential loop wraps EVERYTHING for one player
# -- cache read, fetch, write, prune, fallback -- in one try/except, so a
# broken cache entry fails that player and the run moves on.

def _write_corrupt(cache, slug, age):
    from fpl.data.cache import TS_FMT
    stamp = (now() - age).strftime(TS_FMT)
    (cache.root / f"{slug}_{stamp}.json").write_text("{not json")


def test_a_corrupt_cached_snapshot_fails_only_that_player(tmp_path):
    cache = Cache(tmp_path)
    _write_corrupt(cache, "element-summary-1", timedelta(hours=1))   # fresh, unreadable
    s = PerUrlSession({_es(2): {"id": 2}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2])

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}
    assert c.stale is True
    assert s.calls == [_es(2)]            # today: marked failed, not refetched


def test_a_failed_cache_write_fails_only_that_player(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    real_put = cache.put

    def put(slug, payload, *a, **k):
        if slug == "element-summary-1":
            raise OSError("disk full")
        return real_put(slug, payload, *a, **k)

    monkeypatch.setattr(cache, "put", put)
    s = PerUrlSession({_es(1): {"id": 1}, _es(2): {"id": 2}})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2])

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}         # today: not rescued by a fallback
    assert c.stale is True
    assert sorted(s.calls) == [_es(1), _es(2)]


def test_a_corrupt_fallback_snapshot_fails_only_that_player(tmp_path):
    cache = Cache(tmp_path)
    _write_corrupt(cache, "element-summary-1", timedelta(days=11))   # stale AND unreadable
    s = PerUrlSession({_es(2): {"id": 2}}, failing=[_es(1)])
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2], not_before=now() - timedelta(days=2))

    assert out == {2: {"id": 2}}
    assert c.fetch_failures == {1}


def test_a_raising_progress_callback_still_stops_the_fetch(tmp_path):
    s = PerUrlSession({_es(p): {"id": p} for p in (1, 2, 3)})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    def progress(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        c.element_summaries([1, 2, 3], progress=progress)
```

- [ ] **Step 2: Replace the implementation-coupled test in `tests/test_coverage_gate.py`**

Replace the whole `test_the_client_records_which_players_failed` function (lines 69–84) with:

```python
def test_the_client_records_which_players_failed(tmp_path):
    """`stale` is a single global boolean; it cannot say who is affected."""
    from fpl.data.cache import Cache
    from fpl.data.client import BASE, FplClient

    class Flaky:
        def get(self, url, timeout=None):
            if url == BASE + "element-summary/3/":
                raise RuntimeError("503")

            class R:
                status_code = 200

                def json(self):
                    return {"history_past": [], "history": []}

                def raise_for_status(self):
                    pass
            return R()

    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=Flaky())
    got = client.element_summaries([1, 2, 3])
    assert set(got) == {1, 2}
    assert client.fetch_failures == {3}
```

- [ ] **Step 3: Run the new and changed tests on today's code**

Run: `python -m pytest tests/test_client.py tests/test_coverage_gate.py -v -p no:cacheprovider`
Expected: all PASS. These describe current behaviour; a failure here means the test is wrong, not the code.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q -p no:cacheprovider`
Expected: `784 passed, 4 xfailed` (774 + 10 new parity and containment tests; the coverage-gate test was replaced, not added).

- [ ] **Step 5: Commit**

```bash
git add tests/test_client.py tests/test_coverage_gate.py
git commit -m "test: pin element_summaries semantics before the fetch goes concurrent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Phase 1 — cache index

**Files:**
- Modify: `fpl/data/cache.py:1-4` (imports), `:116-122` (`__init__`, `_paths`), `:124-137` (`put`), `:139-145` (`newest`), `:200-207` (`prune`)
- Test: `tests/test_cache.py` (append), `tests/test_perf_contracts.py` (remove one xfail)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Cache` keeps its public API (`put`, `newest`, `newest_stamp`, `newest_meta`, `get_fresh`, `prune`) and adds `Cache.refresh() -> None`. The private attributes are `_index: dict[str, list[Path]] | None` and `_index_mtime: int | None` (the directory's `st_mtime_ns` when the index was last known to be current). The index is built with `os.scandir(self.root)`, called through the `os` module attribute, which is what the scan-count contract and the rescan test patch.

- [ ] **Step 1: Write the index tests (append to `tests/test_cache.py`)**

```python
# --- the slug index ---------------------------------------------------------
import os
from pathlib import Path


def test_put_after_the_index_is_built_is_visible(tmp_path):
    c = Cache(tmp_path)
    assert c.newest("fixtures") is None          # builds the (empty) index
    c.put("fixtures", [1], now=NOW - timedelta(hours=2))
    c.put("fixtures", [2], now=NOW - timedelta(hours=1))
    assert c.newest("fixtures")[0] == [2]


def test_prune_updates_the_index(tmp_path):
    c = Cache(tmp_path)
    for h in range(5):
        c.put("bootstrap-static", {"n": h}, now=NOW - timedelta(hours=h))
    c.prune("bootstrap-static", keep=3)
    assert len(c._paths("bootstrap-static")) == 3
    assert all(p.exists() for p in c._paths("bootstrap-static"))
    c.put("bootstrap-static", {"n": -1}, now=NOW + timedelta(hours=1))
    assert c.newest("bootstrap-static")[0] == {"n": -1}


def test_index_ignores_files_that_are_not_snapshots(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1], now=NOW, meta={"final_through": 3})
    (tmp_path / "fixtures_notes.json").write_text("{}")       # no timestamp
    (tmp_path / "fixtures_20260101T000000Z.txt").write_text("x")
    fresh = Cache(tmp_path)                                    # index built from disk
    assert fresh.newest("fixtures")[0] == [1]
    assert len(fresh._paths("fixtures")) == 1


def test_index_keeps_prefix_sharing_slugs_apart(tmp_path):
    c = Cache(tmp_path)
    c.put("element-summary-1", {"id": 1}, now=NOW)
    c.put("element-summary-10", {"id": 10}, now=NOW + timedelta(hours=1))
    fresh = Cache(tmp_path)
    assert fresh.newest("element-summary-1")[0] == {"id": 1}
    assert fresh.newest("element-summary-10")[0] == {"id": 10}


def _set_dir_mtime(path, mtime_ns):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, mtime_ns))


def test_another_writer_is_noticed(tmp_path):
    reader, writer = Cache(tmp_path), Cache(tmp_path)
    assert reader.newest("fixtures") is None                   # index built
    writer.put("fixtures", [9], now=NOW)
    # The filesystem normally moves the directory mtime on create; pin a
    # distinct value so the test does not depend on timestamp granularity.
    _set_dir_mtime(tmp_path, os.stat(tmp_path).st_mtime_ns + 10**9)
    assert reader.newest("fixtures")[0] == [9]


def test_a_snapshot_deleted_elsewhere_heals_instead_of_crashing(tmp_path):
    reader = Cache(tmp_path)
    reader.put("fixtures", ["old"], now=NOW - timedelta(hours=1))
    reader.put("fixtures", ["new"], now=NOW)
    assert reader.newest("fixtures")[0] == ["new"]             # index current
    known = os.stat(tmp_path).st_mtime_ns
    reader._paths("fixtures")[0].unlink()                      # "another process" prunes it
    # Worst case: the deletion landed inside one timestamp tick, so the
    # directory mtime looks unchanged and only the missing file tells.
    _set_dir_mtime(tmp_path, known)
    assert reader.newest("fixtures")[0] == ["old"]


def test_own_writes_do_not_trigger_rescans(tmp_path, monkeypatch):
    c = Cache(tmp_path)
    c.newest("fixtures")                                       # index built
    scans = {"n": 0}
    real_scandir, real_glob = os.scandir, Path.glob

    # Both, because pathlib's glob does not list through os.scandir.
    def counting_scandir(path=".", *a, **k):
        scans["n"] += 1
        return real_scandir(path, *a, **k)

    def counting_glob(self, pattern, *a, **k):
        scans["n"] += 1
        return real_glob(self, pattern, *a, **k)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    monkeypatch.setattr(Path, "glob", counting_glob)
    for h in range(20):
        c.put("fixtures", [h], now=NOW + timedelta(hours=h), meta={"final_through": 1})
        c.prune("fixtures", keep=3)
        c.newest("fixtures")
    assert scans["n"] == 0
    assert c.newest("fixtures")[0] == [19]


def test_refresh_sees_another_writer(tmp_path):
    reader, writer = Cache(tmp_path), Cache(tmp_path)
    assert reader.newest("fixtures") is None
    writer.put("fixtures", [9], now=NOW)
    reader.refresh()
    assert reader.newest("fixtures")[0] == [9]
```

- [ ] **Step 2: Run them to see which fail**

Run: `python -m pytest tests/test_cache.py -v -p no:cacheprovider -k "index or refresh or prune_updates or put_after or writer or elsewhere or rescans"`
Expected, on the glob code:
- `test_index_ignores_files_that_are_not_snapshots` FAILS: `fixtures_notes.json` sorts above the real snapshot and `newest` raises `ValueError` from `strptime`.
- `test_refresh_sees_another_writer` FAILS with `AttributeError: 'Cache' object has no attribute 'refresh'`.
- `test_own_writes_do_not_trigger_rescans` FAILS: `glob` lists the directory on every call.
- The other five pass. They're regression guards the index must keep passing.

- [ ] **Step 3: Implement the index in `fpl/data/cache.py`**

Add `import os` next to the existing imports (lines 2–4):

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
```

Replace `__init__` and `_paths` (lines 116–122) with:

```python
class Cache:
    """Timestamped JSON snapshots, one directory, one writer at a time.

    Lookups read an index of the directory instead of listing it: globbing per
    lookup listed the whole cache for every call, and a weekly run makes ~2,700
    of them against ~4,000 files -- 22 of its 40 seconds. The index is checked
    against the directory's mtime on every lookup, so a snapshot written by
    another process is normally noticed; because filesystem timestamps are
    tick-granular that check is best-effort, and a snapshot deleted elsewhere
    is caught when reading it fails (see `newest`).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # slug -> snapshot paths, newest first.
        self._index: dict[str, list[Path]] | None = None
        # The directory mtime at which the index was last known to be current.
        self._index_mtime: int | None = None

    def refresh(self) -> None:
        """Forget the index so the next lookup rescans the directory."""
        self._index = None

    def _dir_mtime(self) -> int:
        return os.stat(self.root).st_mtime_ns

    @staticmethod
    def _slug_of(name: str) -> str | None:
        """The slug a snapshot filename belongs to, or None if it is not one.

        Stricter than the glob it replaces: `fixtures_notes.json` matched
        `fixtures_*.json`, sorted above every real snapshot and crashed
        `newest` on the timestamp parse.
        """
        if not name.endswith(".json"):
            return None
        slug, sep, stamp = name[:-len(".json")].rpartition("_")
        if not sep or not slug:
            return None
        try:
            datetime.strptime(stamp, TS_FMT)
        except ValueError:
            return None
        return slug

    def _current_index(self) -> dict[str, list[Path]]:
        mtime = self._dir_mtime()
        if self._index is None or mtime != self._index_mtime:
            # mtime is read BEFORE the scan: a write that lands during it moves
            # the mtime again and forces another rebuild, never a missed file.
            self._index_mtime = mtime
            index: dict[str, list[Path]] = {}
            with os.scandir(self.root) as entries:
                for entry in entries:
                    slug = self._slug_of(entry.name)
                    if slug is not None and entry.is_file():
                        index.setdefault(slug, []).append(self.root / entry.name)
            for paths in index.values():
                paths.sort(reverse=True)
            self._index = index
        return self._index

    def _paths(self, slug: str) -> list[Path]:
        return list(self._current_index().get(slug, ()))
```

Replace `put` (lines 124–137), keeping its docstring and changing its last sentence as shown:

```python
    def put(self, slug: str, payload, now: datetime | None = None,
            meta: dict | None = None) -> Path:
        """Write a snapshot, and beside it what was true of the data when it was
        taken (`meta`), which no later reader can work out from the timestamp.

        The sidecar deliberately does not end in `.json`: the index only takes
        `*.json` snapshots and would otherwise try to read it as one.
        """
        # Was the index current just before this write? Only then can our own
        # write be folded in; otherwise someone else wrote too, so rebuild.
        current = self._index is not None and self._dir_mtime() == self._index_mtime
        ts = _as_utc(now or _now())
        p = self.root / f"{slug}_{ts.strftime(TS_FMT)}.json"
        p.write_text(json.dumps(payload))
        if meta:
            p.with_suffix(META_SUFFIX).write_text(json.dumps(meta))
        if current:
            paths = self._index.setdefault(slug, [])
            if p not in paths:
                paths.append(p)
                paths.sort(reverse=True)
            self._index_mtime = self._dir_mtime()
        else:
            self._index = None
        return p
```

Replace `newest` (lines 139–145) with:

```python
    def newest(self, slug: str):
        for retried in (False, True):
            paths = self._paths(slug)
            if not paths:
                return None
            p = paths[0]
            try:
                text = p.read_text()
            except FileNotFoundError:
                # Pruned by another process inside one mtime tick, so the index
                # still lists it. Rescan once rather than crash the run.
                if retried:
                    raise
                self.refresh()
                continue
            ts = datetime.strptime(p.stem.rsplit("_", 1)[1], TS_FMT).replace(tzinfo=timezone.utc)
            return json.loads(text), ts
```

Replace `prune` (lines 200–207) with:

```python
    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)          # validates the index against the mtime
        removed = 0
        for p in paths[keep:]:
            # missing_ok: another process may have pruned the same snapshot.
            p.unlink(missing_ok=True)
            p.with_suffix(META_SUFFIX).unlink(missing_ok=True)
            removed += 1
        if removed and self._index is not None:
            self._index[slug] = paths[:keep]
            self._index_mtime = self._dir_mtime()
        return removed
```

- [ ] **Step 4: Run the cache tests and the scan contract**

Remove the `@pytest.mark.xfail(...)` line above `test_cache_scans_directory_once` in `tests/test_perf_contracts.py`.

Run: `python -m pytest tests/test_cache.py tests/test_perf_contracts.py -v -p no:cacheprovider`
Expected: every cache test PASSES; `test_cache_scans_directory_once` PASSES; the three phase 2 contracts still xfail.

- [ ] **Step 5: Full suite, golden check, bench**

```bash
python -m pytest -q -p no:cacheprovider        # expect 793 passed, 3 xfailed
python scripts/bench.py golden check           # expect golden: OK
python scripts/bench.py run --label phase1-cache-index
```
Expected bench: `cache_hits_s` under 1 s (the per-lookup `os.stat` costs about 2,700 × ~20 µs), and `gw6_cache_only_s` ≤ 20 (success criterion 2). `refresh` is still about 650 s extrapolated; that's phase 2. Fill the `phase1-cache-index` column in `docs/perf/README.md`.

- [ ] **Step 6: Commit**

```bash
git add fpl/data/cache.py tests/test_cache.py tests/test_perf_contracts.py docs/perf
git commit -m "perf: index cache snapshots by slug instead of globbing per lookup

One directory scan per Cache instead of ~2,700 on a weekly run. The index
is checked against the directory mtime, so other writers are noticed, and a
snapshot deleted elsewhere triggers one rescan instead of a crash.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Phase 2a — `TokenBucket` rate limiter

**Files:**
- Create: `fpl/data/throttle.py`
- Test: `tests/test_throttle.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FetchCancelled(Exception)`.
  - `TokenBucket(rate: float, clock=time.monotonic, sleep=time.sleep)` with:
    - `acquire(cancel: threading.Event | None = None) -> None`: blocks until this caller may start a request. It raises `FetchCancelled` if `cancel` is set, whether before or while waiting.
    - `pause_for(seconds: float) -> None`: no acquisition by any thread, *including threads already waiting*, before now + seconds.
    - `rate` (float attribute).
  - `rate <= 0` disables spacing, but pauses and cancellation still apply. Thread-safe. Task 6 uses exactly these names.

- [ ] **Step 1: Write the failing tests (`tests/test_throttle.py`)**

```python
import threading
import time

import pytest

from fpl.data.throttle import FetchCancelled, TokenBucket


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(round(s, 9))
        self.t += s


def test_spaces_acquisitions_at_the_rate():
    clk = FakeClock()
    b = TokenBucket(5, clock=clk, sleep=clk.sleep)
    for _ in range(3):
        b.acquire()
    assert clk.sleeps == [0.2, 0.2]


def test_zero_rate_never_waits():
    clk = FakeClock()
    b = TokenBucket(0, clock=clk, sleep=clk.sleep)
    for _ in range(50):
        b.acquire()
    assert clk.sleeps == []


def test_an_idle_bucket_does_not_bank_a_burst():
    clk = FakeClock()
    b = TokenBucket(5, clock=clk, sleep=clk.sleep)
    b.acquire()
    clk.t = 10.0
    b.acquire()
    b.acquire()
    assert clk.sleeps == [0.2]


def test_pause_holds_every_caller_even_uncapped():
    clk = FakeClock()
    b = TokenBucket(0, clock=clk, sleep=clk.sleep)
    b.pause_for(3.0)
    b.acquire()
    assert clk.sleeps == [3.0]


def test_a_pause_holds_back_threads_already_waiting():
    """The review's case: a 429 lands while other workers are already asleep
    waiting for their slot. None of them may start inside the pause."""
    b = TokenBucket(10)
    lock = threading.Lock()
    starts: list[float] = []
    paused: dict = {}

    def worker():
        for _ in range(3):
            b.acquire()
            # Stamp before taking the recording lock: a thread that acquired
            # just before the pause but queued on the lock must not look late.
            started = time.monotonic()
            with lock:
                starts.append(started)
                if len(starts) == 2:
                    b.pause_for(0.5)
                    paused["at"] = time.monotonic()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(starts) == 12
    later = [t for t in starts if t > paused["at"]]
    assert later, "nothing started after the pause"
    assert min(later) >= paused["at"] + 0.5 - 0.02


def test_cancel_wakes_a_waiting_caller():
    b = TokenBucket(0)
    b.pause_for(30.0)
    stop = threading.Event()
    outcome: dict = {}

    def waiter():
        try:
            b.acquire(stop)
            outcome["result"] = "acquired"
        except FetchCancelled:
            outcome["result"] = "cancelled"

    t = threading.Thread(target=waiter)
    started = time.monotonic()
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert outcome["result"] == "cancelled"
    assert time.monotonic() - started < 1.0


def test_a_cancelled_caller_never_acquires():
    stop = threading.Event()
    stop.set()
    with pytest.raises(FetchCancelled):
        TokenBucket(0).acquire(stop)


def test_is_safe_across_threads():
    b = TokenBucket(20)
    t = time.monotonic()
    workers = [threading.Thread(target=lambda: [b.acquire() for _ in range(5)])
               for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # 20 acquisitions at 20/s: the last starts no earlier than 19/20 s in.
    assert time.monotonic() - t >= 0.9
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `python -m pytest tests/test_throttle.py -v -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.data.throttle'`.

- [ ] **Step 3: Implement `fpl/data/throttle.py`**

```python
"""A rate limit shared by every thread fetching from the FPL API."""
import threading
import time


class FetchCancelled(Exception):
    """The fetch this caller belongs to was stopped while it waited."""


class TokenBucket:
    """At most `rate` acquisitions per second across all threads sharing it.

    Capacity is one: acquisitions are spaced 1/rate apart and an idle bucket
    banks nothing, so a pause never turns into a burst -- FPL's edge answers
    bursts with 429s. `rate <= 0` disables the spacing (tests, and anyone who
    explicitly opts out) but still honours pauses and cancellation.

    Nothing is reserved while waiting. Every wake-up re-reads the next free
    slot and the pause under the lock, so a pause set by one worker after a
    429 holds back workers that were already asleep waiting for their turn.
    Booking a future slot and sleeping towards it would let them start inside
    the pause.
    """

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(rate)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0
        self._paused_until = 0.0

    def pause_for(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, self._clock() + float(seconds))

    def acquire(self, cancel: threading.Event | None = None) -> None:
        while True:
            if cancel is not None and cancel.is_set():
                raise FetchCancelled
            with self._lock:
                now = self._clock()
                ready_at = max(self._next, self._paused_until)
                if now >= ready_at:
                    # Spaced from when this caller actually goes, not from
                    # when it was due, so a late wake-up never shortens the gap.
                    if self.rate > 0:
                        self._next = now + 1.0 / self.rate
                    return
                wait = ready_at - now
            if cancel is not None:
                cancel.wait(wait)      # returns early when the fetch is stopped
            else:
                self._sleep(wait)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_throttle.py -v -p no:cacheprovider`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add fpl/data/throttle.py tests/test_throttle.py
git commit -m "feat: cancellable token-bucket limiter whose pauses hold waiting threads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Phase 2b — concurrent `element_summaries`, retries, config, call sites

**Files:**
- Replace: `fpl/data/client.py` (the full new module is below; `_get`, the single-endpoint methods and the throttle keep their behaviour)
- Modify: `fpl/config.py:44-45` (dataclass fields) and `:119-120` (loader), `config.yaml` (`data:` block)
- Modify: `run_gameweek.py:96` and `:109-111` (progress comment), `fpl/pipeline.py:577`, `scripts/run_backtest.py:176`, `scripts/score_gameweek.py:60`
- Test: `tests/test_client.py` (append), `tests/test_config.py` (append), `tests/test_perf_contracts.py` (remove three xfails)

**Interfaces:**
- Consumes: `TokenBucket(rate)`, `.acquire(cancel)`, `.pause_for(seconds)` and `FetchCancelled` from Task 5; the Task 3 parity tests as the behaviour contract.
- Produces:
  - `FplClient(cache, ttl_hours=6, rate_limit_s=1.0, session=None, fetch_workers=4, fetch_rate_per_s=None)`. When `fetch_rate_per_s` is `None`, it resolves to `0.0` if `rate_limit_s` is falsy, else `DEFAULT_FETCH_RATE = 5.0`.
  - Module constants `RETRY_STATUSES`, `MAX_RETRIES = 3`, `BACKOFF_S = (1.0, 2.0, 4.0)`, `MAX_RETRY_AFTER_S = 120.0` and `POLL_S = 0.25`. Tests monkeypatch `BACKOFF_S`.
  - `ServerBackoff(Exception)`.
  - `FplClient._retry_after(resp) -> float | None` (staticmethod).
  - `Config.fetch_workers: int = 4` and `Config.fetch_rate_per_s: float = 5.0`.

**How stopping works.** Read this before the code.
- `_fetch_misses` keeps at most `2 × fetch_workers` futures queued.
- Every worker checks one `threading.Event` before each rate wait, request, backoff and retry. The rate and backoff waits are `Event.wait`, so they wake at once.
- The calling thread waits in `POLL_S` slices, so Ctrl-C is seen on Windows.
- On any exception it sets the event, then **joins** the pool (`shutdown(wait=True, cancel_futures=True)`), and only then closes the worker sessions. No request starts after `element_summaries` has left, and no session is closed under a running request.
- **Stopping is bounded, not instant.** A request already on the wire can't be interrupted, so the join waits for the slowest one in flight: normally a fraction of a second, at worst its 30 s timeout.
- **Per-player containment.** Every cache operation for one player (the hit check, the write and prune, the fallback read) is wrapped so that a failure marks only that player failed. This matches the sequential loop's per-player `try/except`: a corrupt fresh snapshot fails the player without a refetch, and a failed write isn't rescued by a fallback. `progress` is called outside the containment, so a raising callback still stops the whole fetch.
- A `ServerBackoff` (Retry-After > 120 s) sets the same event but doesn't raise. The players not yet fetched take the stale-fallback / `fetch_failures` path.

- [ ] **Step 1: Write the failing retry, Retry-After, interrupt and dedupe tests (append to `tests/test_client.py`)**

```python
# --- concurrent fetch: retries, Retry-After, interruption, duplicates -----------
# The three intentional differences from the sequential crawl live here: a
# transient failure is retried before it counts, an id given twice is fetched
# once, and a Retry-After too long to sit out ends the fetch for this run. Everything else is pinned by the parity tests above.
import threading
import time
from email.utils import format_datetime

import requests

import fpl.data.client as client_mod


class ScriptedResponse:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers = payload, status, headers or {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class ScriptedSession:
    """Each URL answers from its own script of steps; the last step repeats.
    A step is a status code, a (status, headers) pair, or an exception class."""

    def __init__(self, scripts, latency_s: float = 0.0):
        self.scripts = {u: list(s) for u, s in scripts.items()}
        self.latency_s = latency_s
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def get(self, url, timeout=None):
        with self.lock:
            self.calls.append(url)
            steps = self.scripts[url]
            step = steps.pop(0) if len(steps) > 1 else steps[0]
        if self.latency_s:
            time.sleep(self.latency_s)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted")
        status, headers = step if isinstance(step, tuple) else (step, {})
        return ScriptedResponse({"url": url}, status, headers)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(client_mod, "BACKOFF_S", (0.0, 0.0, 0.0))


def test_transient_503_is_retried_until_it_succeeds(tmp_path, no_backoff):
    s = ScriptedSession({_es(1): [503, 503, 200]})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    out = c.element_summaries([1])

    assert out == {1: {"url": _es(1)}}
    assert len(s.calls) == 3
    assert c.fetch_failures == set() and c.stale is False


def test_429_pauses_the_limiter_and_retries(tmp_path, no_backoff, monkeypatch):
    s = ScriptedSession({_es(1): [(429, {"Retry-After": "0"}), 200]})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    pauses = []
    real = c._limiter.pause_for
    monkeypatch.setattr(c._limiter, "pause_for", lambda sec: (pauses.append(sec), real(sec)))

    out = c.element_summaries([1])

    assert 1 in out and len(s.calls) == 2
    assert pauses == [0.0]


def test_connection_errors_are_retried(tmp_path, no_backoff):
    s = ScriptedSession({_es(1): [requests.ConnectionError, 200]})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    assert 1 in c.element_summaries([1])
    assert len(s.calls) == 2


def test_404_is_not_retried(tmp_path, no_backoff):
    s = ScriptedSession({_es(1): [404]})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    assert c.element_summaries([1]) == {}
    assert len(s.calls) == 1
    assert c.fetch_failures == {1}


def test_persistent_503_gives_up_after_bounded_retries(tmp_path, no_backoff):
    cache = Cache(tmp_path)
    cache.put("element-summary-2", {"src": "old"}, now=now() - timedelta(days=11))
    s = ScriptedSession({_es(1): [503], _es(2): [503]})
    c = FplClient(cache, rate_limit_s=0, session=s)

    out = c.element_summaries([1, 2], not_before=now() - timedelta(days=2))

    assert out == {2: {"src": "old"}}                        # stale fallback kept
    assert c.fetch_failures == {1}
    assert s.calls.count(_es(1)) == client_mod.MAX_RETRIES + 1


def test_an_unexpected_error_is_not_retried(tmp_path, no_backoff):
    s = ScriptedSession({_es(1): [RuntimeError]})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)

    assert c.element_summaries([1]) == {}
    assert len(s.calls) == 1


def test_retry_after_accepts_seconds_and_http_dates():
    ra = FplClient._retry_after
    assert ra(ScriptedResponse({}, 429, {"Retry-After": "7"})) == 7.0
    soon = datetime.now(timezone.utc) + timedelta(seconds=90)
    got = ra(ScriptedResponse({}, 429, {"Retry-After": format_datetime(soon, usegmt=True)}))
    assert 85 <= got <= 91
    past = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert ra(ScriptedResponse({}, 429, {"Retry-After": format_datetime(past, usegmt=True)})) == 0.0
    assert ra(ScriptedResponse({}, 429, {"Retry-After": "soon"})) is None
    assert ra(ScriptedResponse({}, 429, {})) is None


def test_a_long_retry_after_stops_asking_and_falls_back(tmp_path, no_backoff):
    cache = Cache(tmp_path)
    for pid in (2, 3, 4, 5):
        cache.put(f"element-summary-{pid}", {"src": "old"}, now=now() - timedelta(days=11))
    scripts = {_es(1): [(429, {"Retry-After": "3600"})]}
    scripts.update({_es(p): [200] for p in (2, 3, 4, 5)})
    s = ScriptedSession(scripts)
    c = FplClient(cache, rate_limit_s=0, session=s, fetch_workers=1)

    out = c.element_summaries([1, 2, 3, 4, 5], not_before=now() - timedelta(days=2))

    assert s.calls.count(_es(1)) == 1                  # never re-asked
    assert len(s.calls) <= 2                           # at most the one already queued
    assert set(out) == {2, 3, 4, 5}
    assert all(out[p] == {"src": "old"} for p in (3, 4, 5))
    assert c.fetch_failures == {1}
    assert c.stale is True


class _Stop(Exception):
    pass


def _stop_at(n):
    def progress(done, total):
        if done == n:
            raise _Stop
    return progress


def test_an_interrupted_refresh_stops_its_workers(tmp_path):
    s = ScriptedSession({_es(p): [200] for p in range(1, 41)}, latency_s=0.1)
    cache = Cache(tmp_path)
    c = FplClient(cache, rate_limit_s=0, session=s, fetch_workers=2, fetch_rate_per_s=0)

    t = time.monotonic()
    with pytest.raises(_Stop):
        c.element_summaries(range(1, 41), progress=_stop_at(3))
    elapsed = time.monotonic() - t
    calls_at_return = len(s.calls)
    time.sleep(0.3)

    assert elapsed < 1.0                               # draining would take 2 s
    assert len(s.calls) == calls_at_return             # no worker is still fetching
    assert calls_at_return <= 3 + 2 * 2                # only the bounded window ran
    assert sum(1 for p in range(1, 41) if cache.newest(f"element-summary-{p}")) >= 3


def test_an_interrupt_cuts_a_retry_backoff_short(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "BACKOFF_S", (5.0, 5.0, 5.0))
    scripts = {_es(1): [503, 200]}
    scripts.update({_es(p): [200] for p in range(2, 9)})
    s = ScriptedSession(scripts, latency_s=0.02)
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s,
                  fetch_workers=2, fetch_rate_per_s=0)

    t = time.monotonic()
    with pytest.raises(_Stop):
        c.element_summaries(range(1, 9), progress=_stop_at(3))

    assert time.monotonic() - t < 1.0                  # not the 5 s backoff
    assert s.calls.count(_es(1)) == 1                  # and no retry after the stop


def test_duplicate_ids_are_fetched_once(tmp_path):
    s = PerUrlSession({_es(1): {"id": 1}, _es(2): {"id": 2}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    seen = []

    out = c.element_summaries([1, 2, 1], progress=lambda d, t: seen.append((d, t)))

    assert list(out) == [1, 2]
    assert sorted(s.calls) == [_es(1), _es(2)]
    assert seen[-1] == (2, 2)


def test_rate_limit_zero_leaves_the_fetch_uncapped(tmp_path):
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=PerUrlSession({}))
    assert c._limiter.rate == 0.0
    assert FplClient(Cache(tmp_path), session=PerUrlSession({}))._limiter.rate == 5.0
    assert FplClient(Cache(tmp_path), rate_limit_s=0, session=PerUrlSession({}),
                     fetch_rate_per_s=2)._limiter.rate == 2.0
```

- [ ] **Step 2: Write the failing config tests (append to `tests/test_config.py`)**

```python
def test_fetch_settings_load_from_the_data_block(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("data:\n  fetch_workers: 2\n  fetch_rate_per_s: 3.5\n")
    c = load_config(p)
    assert c.fetch_workers == 2
    assert c.fetch_rate_per_s == 3.5


def test_fetch_settings_default_when_absent(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("budget: 100.0\n")
    c = load_config(p)
    assert c.fetch_workers == 4
    assert c.fetch_rate_per_s == 5.0
```

- [ ] **Step 3: Run the new tests to confirm they fail**

Run: `python -m pytest tests/test_client.py tests/test_config.py -v -p no:cacheprovider`
Expected: most Step 1 tests FAIL or ERROR, because the `no_backoff` fixture can't patch the missing `BACKOFF_S`; `_limiter`, `_retry_after` and `fetch_workers` don't exist; and the old code neither retries nor dedupes. `test_404_is_not_retried` and `test_an_unexpected_error_is_not_retried` may already pass (the old code never retries anything). The Step 2 tests FAIL with `AttributeError: 'Config' object has no attribute 'fetch_workers'`. Every Task 3 parity test still PASSES.

- [ ] **Step 4: Replace `fpl/data/client.py` with this module**

It was prototyped against every test in this plan, including the Task 3 parity and containment tests (all passing).

```python
"""Read-only HTTP client for the public FPL API.

Never calls any endpoint requiring a login/session cookie (e.g. my-team/),
and never issues a non-GET request.
"""
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from .cache import Cache
from .throttle import FetchCancelled, TokenBucket

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; fpl-team-picker/1.0)"}
FORBIDDEN = ("my-team",)
# `history_past` for a completed season is immutable, so these can cache hard.
HISTORY_TTL_H = 24 * 30
# Element-summaries are the one bulk fetch: ~667 of them after every gameweek.
# One at a time behind a 1 s throttle that was 11+ minutes. A few in flight
# under one shared cap overlaps the network latency without asking FPL for
# more than a handful of requests a second.
DEFAULT_FETCH_RATE = 5.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BACKOFF_S = (1.0, 2.0, 4.0)
# The longest server-requested wait a run sits out. A longer Retry-After is
# obeyed by not asking again this run: the players still to fetch fall back to
# their cached snapshots, exactly as they would in an outage.
MAX_RETRY_AFTER_S = 120.0
# How often the calling thread wakes while workers fetch. A blocking wait on
# Windows does not see Ctrl-C until it returns; a short timed one does.
POLL_S = 0.25


class DataCoverageError(RuntimeError):
    """Too much of this run's player history is missing to optimise safely.

    Raised rather than warned because the failure is invisible downstream: a
    player whose summary could not be fetched is zeroed and routed to the same
    price prior as a genuine new signing, and the optimizer then returns a
    confident, legal team built on an estimate nothing supports.
    """


class ServerBackoff(Exception):
    """FPL asked us to stay away longer than a run can wait."""


class FplClient:
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0,
                 session=None, fetch_workers: int = 4,
                 fetch_rate_per_s: float | None = None):
        self.cache = cache
        self.ttl_hours = ttl_hours
        self.rate_limit_s = rate_limit_s
        self.session = session or requests.Session()
        self.stale = False
        # When each payload this run used was captured. A forecast is only
        # reproducible if it records which snapshots it read: "the model got
        # worse" and "the model was handed a week-old bootstrap" look identical
        # in the score ledger otherwise.
        self.sources: dict[str, str] = {}
        # Recorded alongside every snapshot this client writes. `final_through`
        # says which gameweeks FPL had finished CHECKING at capture time, which
        # a timestamp cannot express and a later reader cannot recover.
        self.snapshot_meta: dict = {}
        # Slugs served from a snapshot too old to carry a `final_through`
        # marker, when one was asked for.
        self.unverified: set[str] = set()
        # Players whose element-summary could not be fetched OR served from
        # cache on this run. `stale` is one global boolean and cannot say WHO
        # is affected, which is exactly what the caller needs to decide whether
        # the missing history touches the squad.
        self.fetch_failures: set[int] = set()
        self._last_call = 0.0
        # A caller that turned the throttle off (the tests) has turned the
        # bulk fetch's cap off too, unless it says otherwise.
        if fetch_rate_per_s is None:
            fetch_rate_per_s = DEFAULT_FETCH_RATE if rate_limit_s else 0.0
        self.fetch_workers = max(1, int(fetch_workers))
        self._limiter = TokenBucket(fetch_rate_per_s)
        # An injected session (tests, benchmarks) is shared by every worker. A
        # real one is per thread: requests does not promise a Session is safe
        # to use from several threads at once.
        self._shared_session = session is not None
        self._local = threading.local()
        self._worker_sessions: list = []
        self._sessions_lock = threading.Lock()

    def _record_source(self, slug: str) -> None:
        ts = self.cache.newest_stamp(slug)
        if ts is not None:
            self.sources[slug] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def source_summary(self) -> dict:
        """Snapshot times, with the per-player summaries collapsed to a range.

        Seven hundred element-summary entries say nothing a first-and-last pair
        does not, and would dwarf the forecast they annotate.
        """
        out, summaries = {}, []
        for slug, ts in self.sources.items():
            if slug.startswith("element-summary-"):
                summaries.append(ts)
            else:
                out[slug] = ts
        if summaries:
            out["element-summary"] = {"oldest": min(summaries), "newest": max(summaries),
                                      "n": len(summaries)}
        return out

    def _throttle(self) -> None:
        if self.rate_limit_s:
            delta = time.monotonic() - self._last_call
            if delta < self.rate_limit_s:
                time.sleep(self.rate_limit_s - delta)
            self._last_call = time.monotonic()

    def _cached(self, slug: str, ttl_hours: float, not_before=None,
                require_final_through=None):
        """The fresh cached payload for `slug`, or None -- with the source and
        the `unverified` caveat recorded exactly as a live fetch would."""
        cached = self.cache.get_fresh(slug, ttl_hours, not_before=not_before,
                                      require_final_through=require_final_through)
        if cached is None:
            return None
        self._record_source(slug)
        # Served a snapshot that predates the `final_through` marker, so it
        # cannot say whether it was taken before or after FPL's data check.
        # It is still used -- refusing every pre-marker snapshot would
        # re-fetch 650 players for data that is usually settled -- but the
        # caller is told, so a report can carry the caveat.
        if (require_final_through is not None
                and "final_through" not in self.cache.newest_meta(slug)):
            self.unverified.add(slug)
        return cached

    def _store(self, slug: str, payload) -> None:
        self.cache.put(slug, payload, meta=self.snapshot_meta or None)
        self.cache.prune(slug, keep=3)
        self._record_source(slug)

    def _fallback(self, slug: str):
        """The newest snapshot however old, marking the run stale, or None."""
        got = self.cache.newest(slug)
        if got is None:
            return None
        self.stale = True
        self._record_source(slug)
        return got[0]

    def _get(self, path: str, slug: str, ttl_hours: float | None = None,
             not_before=None, require_final_through=None):
        ttl = self.ttl_hours if ttl_hours is None else ttl_hours
        cached = self._cached(slug, ttl, not_before, require_final_through)
        if cached is not None:
            return cached
        url = BASE + path
        if any(f in url.lower() for f in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        try:
            self._throttle()
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            fallback = self._fallback(slug)
            if fallback is None:
                raise
            return fallback
        self._store(slug, payload)
        return payload

    def _worker_session(self):
        if self._shared_session:
            return self.session
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
            with self._sessions_lock:
                self._worker_sessions.append(s)
        return s

    def _close_worker_sessions(self) -> None:
        with self._sessions_lock:
            sessions, self._worker_sessions = self._worker_sessions, []
        for s in sessions:
            s.close()

    @staticmethod
    def _retry_after(resp) -> float | None:
        """Seconds the server asked us to wait, or None if it did not say.

        RFC 9110 allows either delay-seconds or an HTTP-date.
        """
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        if value is None:
            return None
        value = str(value).strip()
        try:
            return max(float(value), 0.0)
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)

    @staticmethod
    def _pause(seconds: float, stop: threading.Event) -> None:
        if stop.wait(seconds):
            raise FetchCancelled

    def _fetch_json(self, url: str, stop: threading.Event):
        """GET `url` on a worker thread, through the shared limiter.

        Retries what is worth retrying -- rate limiting, server errors, dropped
        connections -- with backoff; a 404 or anything unexpected fails at
        once. Every wait watches `stop`, so a stopped fetch leaves promptly.
        Touches no cache and no client state: the calling thread does all of
        that, so none of it needs a lock.
        """
        if any(f in url.lower() for f in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        session = self._worker_session()
        for attempt in range(MAX_RETRIES + 1):
            last = attempt == MAX_RETRIES
            self._limiter.acquire(stop)
            try:
                resp = session.get(url, timeout=30)
            except (requests.ConnectionError, requests.Timeout):
                if last:
                    raise
                self._pause(BACKOFF_S[attempt], stop)
                continue
            status = getattr(resp, "status_code", 200)
            if status in RETRY_STATUSES:
                asked = self._retry_after(resp)
                if asked is not None and asked > MAX_RETRY_AFTER_S:
                    raise ServerBackoff(f"{url}: Retry-After {asked:.0f}s")
                wait_s = max(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)], asked or 0.0)
                if status == 429:
                    # FPL is telling every worker to slow down, not just this one.
                    self._limiter.pause_for(wait_s)
                if not last:
                    if status != 429:
                        self._pause(wait_s, stop)
                    continue
            resp.raise_for_status()
            return resp.json()

    def bootstrap(self) -> dict:
        return self._get("bootstrap-static/", "bootstrap-static")

    def fixtures(self) -> list[dict]:
        return self._get("fixtures/", "fixtures")

    def element_summary(self, player_id: int, ttl_hours: float | None = None,
                        not_before=None, require_final_through=None) -> dict:
        return self._get(f"element-summary/{player_id}/", f"element-summary-{player_id}",
                         ttl_hours=ttl_hours, not_before=not_before,
                         require_final_through=require_final_through)

    def element_summaries(self, player_ids, ttl_hours: float = HISTORY_TTL_H,
                          progress=None, not_before=None,
                          require_final_through=None) -> dict[int, dict]:
        """Fetch many element-summaries, tolerating individual failures.

        The long default TTL dates from when `history_past` was the only field
        read from these — it is immutable once a season ends, so age did not
        matter. `history_current_frame` now reads THIS season's rounds from the
        same payload, which age very much does affect, so callers that need the
        current season must pass `not_before=data_complete_after(fixtures)`.
        Without it a 30-day-old snapshot counts as fresh and the model silently
        runs on whatever gameweek happened to be current when it was taken.

        A player whose summary can't be fetched is omitted from the result AND
        recorded in `fetch_failures`. Downstream, an omitted player is zeroed
        and routed to the price prior -- the right treatment for a newcomer and
        badly wrong for an established player lost to an outage -- so the
        caller has to be able to tell the two apart.

        Cache hits are resolved first, on this thread. Only the misses go to
        the network, `fetch_workers` at a time under the shared rate cap, and
        each result is written back on this thread as it arrives -- so the
        cache and the bookkeeping above never see two threads at once. A
        transient failure is retried before it counts as one, and an id given
        twice is fetched once.

        Every cache operation for one player -- reading, writing, pruning,
        falling back -- fails THAT player only, exactly as the sequential loop's
        per-player try/except did: a corrupt snapshot or a failed write marks
        the player failed and the run moves on. `progress` stays outside that
        containment, so a raising callback still stops the whole fetch.
        """
        ids = list(dict.fromkeys(int(pid) for pid in player_ids))
        found: dict[int, dict] = {}
        done = 0

        def tick() -> None:
            nonlocal done
            done += 1
            if progress:
                progress(done, len(ids))

        def fail(pid: int) -> None:
            self.stale = True
            self.fetch_failures.add(pid)

        def settle(pid: int, payload, fetched: bool) -> None:
            """Record one player's outcome. `payload` None means the request
            failed; `fetched` means it came off the network and must be stored."""
            slug = f"element-summary-{pid}"
            try:
                if payload is None:
                    payload = self._fallback(slug)
                elif fetched:
                    self._store(slug, payload)
            except Exception:
                # As before: a cache that cannot be read or written fails the
                # player, and a failed write is not rescued by a fallback.
                payload = None
            if payload is None:
                fail(pid)
            else:
                found[pid] = payload
            tick()

        misses = []
        for pid in ids:
            try:
                cached = self._cached(f"element-summary-{pid}", ttl_hours, not_before,
                                      require_final_through)
            except Exception:
                # An unreadable snapshot fails the player without a refetch --
                # what the sequential loop did.
                fail(pid)
                tick()
                continue
            if cached is None:
                misses.append(pid)
            else:
                found[pid] = cached
                tick()

        if misses:
            self._fetch_misses(misses, settle)
        return {pid: found[pid] for pid in ids if pid in found}

    def _fetch_misses(self, misses: list[int], settle) -> None:
        """Fetch `misses` on worker threads and `settle` each on this one.

        At most two requests per worker are queued at any moment, so stopping
        never has hundreds of queued futures to cancel. Stopping -- an
        exception here, including Ctrl-C or a raising progress callback -- sets
        `stop`, which every worker checks before each rate wait, request,
        backoff and retry; the pool is then joined, so no request starts after
        this method has left, and only then are the workers' sessions closed.
        A request already on the wire cannot be interrupted, so stopping takes
        as long as the slowest one in flight -- normally a fraction of a
        second, at worst its 30 s timeout.
        """
        stop = threading.Event()
        queue = iter(misses)
        pending: dict = {}
        pool = ThreadPoolExecutor(max_workers=self.fetch_workers)

        def submit_next() -> None:
            pid = next(queue, None)
            if pid is not None:
                url = f"{BASE}element-summary/{pid}/"
                pending[pool.submit(self._fetch_json, url, stop)] = pid

        try:
            for _ in range(2 * self.fetch_workers):
                submit_next()
            while pending:
                finished, _ = wait(pending, timeout=POLL_S, return_when=FIRST_COMPLETED)
                for fut in finished:
                    pid = pending.pop(fut)
                    try:
                        payload = fut.result()
                    except ServerBackoff:
                        # Obey it: ask for nothing more this run.
                        stop.set()
                        payload = None
                    except Exception:
                        payload = None
                    settle(pid, payload, fetched=payload is not None)
                    if not stop.is_set():
                        submit_next()
            # Only reached with ids left after a ServerBackoff: they were never
            # asked for, and take the same fallback as a failed request.
            for pid in queue:
                settle(pid, None, fetched=False)
        except BaseException:
            stop.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self._close_worker_sessions()

    def entry(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/", f"entry-{entry_id}")

    def entry_history(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/history/", f"entry-history-{entry_id}")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/", f"entry-picks-{entry_id}-{gw}")
```

- [ ] **Step 5: Add the config keys**

`fpl/config.py`, after `cache_ttl_matchday_hours: int = 1` (line 45):

```python
    # The bulk element-summary fetch after each gameweek: requests in flight
    # and the global cap across them. 1 worker at 1.0/s restores the old
    # pacing (retries remain) -- the lever to pull if FPL starts refusing us.
    fetch_workers: int = 4
    fetch_rate_per_s: float = 5.0
```

In `load_config`, after the `cache_ttl_matchday_hours=` line (line 120):

```python
        fetch_workers=int(data.get("fetch_workers", d.fetch_workers)),
        fetch_rate_per_s=float(data.get("fetch_rate_per_s", d.fetch_rate_per_s)),
```

`config.yaml`, in the `data:` block:

```yaml
data:
  cache_ttl_hours: 6
  cache_ttl_matchday_hours: 1
  fetch_workers: 4        # element-summary requests in flight
  fetch_rate_per_s: 5.0   # global cap across workers; 0 = uncapped
```

- [ ] **Step 6: Pass the settings at every production call site**

```python
# run_gameweek.py:96
        client = FplClient(Cache(data_root / "cache"), ttl_hours=cfg.cache_ttl_hours,
                           fetch_workers=cfg.fetch_workers,
                           fetch_rate_per_s=cfg.fetch_rate_per_s)
# fpl/pipeline.py:577
    client = client or FplClient(Cache(root / "cache"), ttl_hours=cfg.cache_ttl_hours,
                                 fetch_workers=cfg.fetch_workers,
                                 fetch_rate_per_s=cfg.fetch_rate_per_s)
# scripts/run_backtest.py:176
    client = FplClient(cache, ttl_hours=cfg.cache_ttl_hours,
                       fetch_workers=cfg.fetch_workers, fetch_rate_per_s=cfg.fetch_rate_per_s)
# scripts/score_gameweek.py:60
    client = FplClient(Cache(DATA_ROOT / "cache"), ttl_hours=ttl,
                       fetch_workers=cfg.fetch_workers, fetch_rate_per_s=cfg.fetch_rate_per_s)
```

In `run_gameweek.py`, the progress comment at lines 110–111 changes from "Player history is fetched one request per second on a cold cache, so a first run takes minutes." to "Player history is fetched a few requests at a time on a cold cache, so a first run still takes a couple of minutes."

Verify that nothing constructs a production client without the settings:
Run: `grep -rn "FplClient(" fpl scripts run_gameweek.py`
Expected: the four lines above, plus the two in `scripts/bench.py`. Both of those pass the settings from `config.yaml` (Task 1) or use no network at all.

- [ ] **Step 7: Remove the three phase 2 xfail markers and run the targeted tests**

Delete the `@pytest.mark.xfail(...)` lines above `test_refresh_is_concurrent`, `test_refresh_respects_worker_bound` and `test_refresh_respects_rate_cap` in `tests/test_perf_contracts.py`.

Run: `python -m pytest tests/test_client.py tests/test_config.py tests/test_perf_contracts.py tests/test_coverage_gate.py tests/test_throttle.py tests/test_cache.py -v -p no:cacheprovider`
Expected: all PASS, including every Task 3 parity test unchanged.

- [ ] **Step 8: Full suite, golden check, bench**

```bash
python -m pytest -q -p no:cacheprovider        # expect 818 passed, 0 xfailed
python scripts/bench.py golden check           # expect golden: OK
python scripts/bench.py run --label phase2-concurrent-fetch
```
Expected bench: `refresh.req_per_s` ≈ 5 and `refresh.extrapolated_667_s` ≈ 135–150 (success criterion 1: ≤ 180). `cache_hits_s` and `gw6_cache_only_s` should be about the same as phase 1. Fill the `phase2-concurrent-fetch` column in `docs/perf/README.md`.

- [ ] **Step 9: Commit**

```bash
git add fpl/data/client.py fpl/config.py config.yaml run_gameweek.py fpl/pipeline.py scripts/run_backtest.py scripts/score_gameweek.py tests/test_client.py tests/test_config.py tests/test_perf_contracts.py docs/perf
git commit -m "perf: fetch element-summaries concurrently under a shared rate cap

4 in flight at 5 req/s: a post-gameweek refresh drops from 11+ minutes to
about 2.5. Transient failures (429/5xx/connection) are retried with backoff
and Retry-After is honoured, never shortened; a wait past 120 s ends the
fetch and the rest fall back to cache. Stopping joins the workers before
returning. Cache writes and failure bookkeeping stay on the calling thread.
fetch_workers: 1 / fetch_rate_per_s: 1.0 restores the old request pacing.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Live smoke test (manual; needs the network; the only step that talks to FPL)**

In a scratch copy of the repo (never the worktree's own `data/`), delete ~50 players' element-summary snapshots so they must be refetched, then run the real pipeline once:

```bash
S=$(mktemp -d) && cp -r fpl run_gameweek.py config.yaml data "$S"/ && cd "$S" \
  && ls data/cache/element-summary-*.json | head -150 | xargs rm \
  && time python run_gameweek.py --mode 2 --gw 6 | tail -5
```
Expected: "Fetching player history" completes ~50 refetches in ≈ 10–15 s, no 429 storms (a stray retry is fine), and a report renders without errors. It will **not** necessarily match the golden decision: without `--no-refresh`, bootstrap and fixtures are refreshed live too. Also press Ctrl-C once during a second such run. It must return to the prompt as soon as the requests already in flight finish (normally well under a second, bounded by the 30 s request timeout), with one traceback, not a storm, and no further "Fetching" lines. If FPL refuses requests, lower `fetch_rate_per_s` in `config.yaml` and report it. Don't raise the default.

**Stop here.** Phases 0–2 are the agreed scope. Report the README table to the user; Task 7 runs only if they say go.

---

### Task 7 (optional, only on an explicit go): Phase 3 — xP inner loop

**Files:**
- Modify: `fpl/model/xp.py:44` (below `MAX_THRESHOLDS`; `numpy as np` is already imported), `:162-178` (`expected_thresholds`), `:289-321` (`xp_for_fixture` signature), `:340-359` (`build_xp` loop)
- Test: `tests/test_xp.py` (append)

**Interfaces:**
- Consumes: the golden check from Task 1.
- Produces: `xp_for_fixture(rate_row, mins_row, fx_row, position, bonus, assist_feasibility_scale: float | None = None) -> float`. `None` means reading `fx_row["assist_feasibility_scale"]` if present, else 1.0, which is today's behaviour, so the other callers are unaffected.

- [ ] **Step 1: Write the parity test for the vectorised tail (append to `tests/test_xp.py`)**

```python
def test_vectorised_threshold_tail_matches_the_series_sum():
    from scipy.stats import poisson
    from fpl.model.xp import MAX_THRESHOLDS, expected_thresholds

    for per_point in (2, 3):
        for lam in (0.01, 0.3, 1.0, 2.7, 6.0, 12.0):
            reference = sum(poisson.sf(per_point * m - 1, lam)
                            for m in range(1, MAX_THRESHOLDS + 1))
            assert abs(expected_thresholds(lam, per_point) - reference) < 1e-12
    assert expected_thresholds(0.0, 3) == 0.0
    assert expected_thresholds(-1.0, 3) == 0.0
```

- [ ] **Step 2: Run it (it passes on today's code, since it's a parity guard)**

Run: `python -m pytest tests/test_xp.py -v -p no:cacheprovider -k vectorised`
Expected: PASS.

- [ ] **Step 3: Vectorise `expected_thresholds`**

Directly below `MAX_THRESHOLDS = 10` add:

```python
_THRESHOLD_STEPS = np.arange(1, MAX_THRESHOLDS + 1)
```

Replace the body of `expected_thresholds` after `if lam <= 0: return 0.0` with:

```python
    # One vectorised sf call instead of MAX_THRESHOLDS scalar ones: scipy's
    # per-call overhead, not the arithmetic, was ~3 s of a weekly run.
    return float(poisson.sf(per_point * _THRESHOLD_STEPS - 1, lam).sum())
```

- [ ] **Step 4: Group fixtures once and drop the per-fixture copy in `build_xp`**

Change the `xp_for_fixture` signature and its assist-feasibility lines:

```python
def xp_for_fixture(rate_row, mins_row, fx_row, position: str, bonus: float,
                   assist_feasibility_scale: float | None = None) -> float:
    ...
    if assist_feasibility_scale is None:
        assist_feasibility_scale = (float(fx_row["assist_feasibility_scale"])
                                    if "assist_feasibility_scale" in fx_row else 1.0)
```

In `build_xp`, before `rows = []`:

```python
    # Each player's fixtures by team, built once: filtering the whole frame
    # per player was O(players x fixtures).
    fixtures_by_team = {int(t): g for t, g in tfx.groupby("team_id")}
    no_fixtures = tfx.iloc[0:0]
```

Replace `fixtures = tfx[tfx["team_id"] == team_id]` with `fixtures = fixtures_by_team.get(team_id, no_fixtures)`. Replace the `fx_for_player = fx.copy()` block and the call below it with:

```python
            feasibility = (assist_feasibility_scales.get(
                (team_id, int(fx["fixture_id"]), pid), 1.0)
                if "fixture_id" in fx else None)
            per_event[event] = per_event.get(event, 0.0) + xp_for_fixture(
                rate_row, mins_row, fx, pos, bonus,
                assist_feasibility_scale=feasibility,
            )
```

- [ ] **Step 5: Tests, golden check, bench**

```bash
python -m pytest tests/test_xp.py -q -p no:cacheprovider
python -m pytest -q -p no:cacheprovider        # expect 819 passed
python scripts/bench.py golden check           # expect golden: OK (xP within 1e-4)
python scripts/bench.py run --label phase3-xp
```
Expected: `gw6_cache_only_s` ≤ 16. Fill the `phase3-xp` column in `docs/perf/README.md`.

- [ ] **Step 6: Commit**

```bash
git add fpl/model/xp.py tests/test_xp.py docs/perf
git commit -m "perf: vectorise the xP threshold tail and group fixtures by team once

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
