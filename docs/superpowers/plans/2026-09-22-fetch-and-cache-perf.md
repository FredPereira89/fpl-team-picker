# Fetch and Cache Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the weekly FPL run's refresh from 11+ minutes to about 2.5 and its cache-only run from 40.6 s to ≤ 20 s, with no change to any recommendation.

**Architecture:** First measure (a bench harness, a frozen GW6 golden output, and complexity tests that fail on today's code). Then replace the per-lookup directory `glob` in `Cache` with a lazily built slug→paths index. Then fetch missing element-summaries through a bounded thread pool behind a shared token-bucket rate limiter with 429/5xx backoff. HTTP runs on worker threads; every cache write and piece of client bookkeeping stays on the main thread. An optional last phase vectorises the xP threshold tail.

**Tech Stack:** Python 3.13, pandas 2.2, numpy 1.26, scipy 1.14, requests 2.31, pytest 8, and `concurrent.futures` / `threading` from the stdlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`

## Global Constraints

- Work only in the worktree `C:\Users\user\Documents\FPL Team Picker\.claude\worktrees\perf-fetch-and-cache` (branch `worktree-perf-fetch-and-cache`). Never run git against the main checkout.
- No new third-party dependencies (`requirements.txt` stays unchanged).
- No behaviour change. `element_summaries` must leave the same dict, `fetch_failures`, `stale`, `unverified` and `sources` as today's code. The GW6 golden decision must stay identical, and every numeric xP column must match within **1e-4 absolute**.
- All existing tests pass after every task (774 at the start; `python -m pytest -q -p no:cacheprovider`).
- Defaults: `fetch_workers: 4`, `fetch_rate_per_s: 5.0`. `fetch_workers: 1` with `fetch_rate_per_s: 1.0` must reproduce today's sequential behaviour.
- Only `FplClient.element_summaries` becomes concurrent. `_get`, `bootstrap`, `fixtures`, `entry*` and the `rate_limit_s` throttle keep their current behaviour.
- All cache and client-state mutation happens on the main thread. Worker threads only perform HTTP.
- Tests assert complexity (scan counts, in-flight counts, rate windows), never absolute wall time on real data.
- The code style matches the repo: comments explain *why*, in full sentences, at the density of the surrounding code.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Ctrl-C (or any exception) in the middle of a 667-player refresh** must stop promptly, not wait for every queued request. Summaries already fetched must be on disk, so a re-run resumes. → Task 6, `test_an_interrupted_refresh_stops_promptly_and_keeps_what_it_fetched`.
2. **FPL answers a burst with 429 (with or without `Retry-After`).** The request is retried, all workers pause together, and the player is not recorded as a failure. → Task 6, `test_429_pauses_the_limiter_and_retries`.
3. **Players whose request keeps failing** (404 for a removed player, persistent 503) end in the stale fallback or in `fetch_failures` after a bounded number of tries. A 404 is never retried and nothing loops forever. → Task 6, `test_404_is_not_retried` and `test_persistent_503_gives_up_after_bounded_retries`.
4. **Cache folder holding files that aren't snapshots**, or slugs sharing a prefix (`.meta` sidecars, a stray `fixtures_notes.json`, `element-summary-1` next to `element-summary-10`): each lookup returns only its own snapshots and never crashes. → Task 4, `test_index_ignores_files_that_are_not_snapshots` and `test_index_keeps_prefix_sharing_slugs_apart`.
5. **Snapshots written or pruned after the index was built** (the refresh writes 667 new ones mid-run) are visible to the next lookup, and pruned ones are gone. → Task 4, `test_put_after_the_index_is_built_is_visible` and `test_prune_updates_the_index`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/bench.py` | create | Benchmarks (`run`) and the GW6 golden capture and check (`golden capture` / `golden check`). All pipeline runs happen in a temporary copy of the repo. |
| `docs/perf/README.md` | create | The before/after table for every phase. |
| `docs/perf/*.json` | create | Raw benchmark output, one file per `bench.py run --label`. |
| `docs/perf/golden-gw6/decision.json`, `xp.parquet` | create | The frozen GW6 decision and xP frame from unmodified master. |
| `tests/test_perf_contracts.py` | create | Complexity contracts: directory scans, in-flight bound, rate cap, real concurrency. |
| `tests/test_client.py` | modify | Parity tests pinning `element_summaries` semantics, plus retry and interrupt tests. |
| `tests/test_coverage_gate.py` | modify | Replace the implementation-coupled client test with a behavioural one. |
| `tests/test_cache.py` | modify | Index behaviour tests. |
| `tests/test_throttle.py` | create | `TokenBucket` unit tests. |
| `tests/test_config.py` | modify | The new `data.fetch_*` keys. |
| `fpl/data/cache.py` | modify | Slug→paths index replacing the per-lookup `glob`. |
| `fpl/data/throttle.py` | create | `TokenBucket`: a thread-safe global rate limiter with a shared pause. |
| `fpl/data/client.py` | modify | Concurrent `element_summaries`, the retrying `_fetch_json`, per-thread sessions, and shared helpers `_cached` / `_store` / `_fallback`. |
| `fpl/config.py`, `config.yaml` | modify | `fetch_workers`, `fetch_rate_per_s`. |
| `run_gameweek.py`, `fpl/pipeline.py`, `scripts/run_backtest.py`, `scripts/score_gameweek.py` | modify | Pass the fetch settings to `FplClient`. |
| `fpl/model/xp.py` | modify (Task 7, optional) | Vectorised threshold tail; per-team fixture grouping; no per-fixture copy. |

---

### Task 1: Freeze inputs, bench harness, golden capture, baseline numbers

**Files:**
- Create: `scripts/bench.py`
- Create: `docs/perf/README.md`, `docs/perf/baseline.json` (generated), `docs/perf/golden-gw6/decision.json` and `xp.parquet` (generated)
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
from fpl.data.cache import Cache  # noqa: E402
from fpl.data.client import FplClient  # noqa: E402

PERF = ROOT / "docs" / "perf"
GOLDEN = PERF / "golden-gw6"
GW = 6
N_PLAYERS = 667
REFRESH_N = 30
REFRESH_LATENCY_S = 0.15
XP_TOL = 1e-4

# Runs inside the temporary repo copy. It wraps `run` so the Recommendation and
# the xP frame can be captured without teaching run_gameweek a new flag.
DRIVER = r'''
import json, sys
import run_gameweek as rg

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


def bench_refresh(n: int = REFRESH_N, latency_s: float = REFRESH_LATENCY_S) -> dict:
    """An empty cache, fetched at the PRODUCTION settings of whichever code is
    checked out (FplClient's own defaults), then extrapolated to 667 players."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        client = FplClient(Cache(Path(d)), session=_LatencySession(latency_s))
        t = time.perf_counter()
        out = client.element_summaries(range(1, n + 1))
        elapsed = time.perf_counter() - t
    assert len(out) == n, "refresh: some fetches failed"
    return {"n": n, "latency_s": latency_s, "seconds": round(elapsed, 3),
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


def _golden_run() -> tuple[dict, pd.DataFrame]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = _temp_repo(Path(d))
        (repo / "golden_driver.py").write_text(DRIVER)
        proc = subprocess.run([sys.executable, "golden_driver.py", str(GW)], cwd=repo,
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
    decision, xp = _golden_run()
    if action == "capture":
        GOLDEN.mkdir(parents=True, exist_ok=True)
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
Expected: `capture` prints the decision (GW6 squad after the Wildcard, captain, and so on). `check` prints `golden: OK` and exits 0. If `check` reports drift on unchanged code, **stop**: the run isn't deterministic (look for simulation seeds or time-dependent inputs), and no later parity claim means anything until that's fixed.

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
Expected: `780 passed, 4 xfailed` (774 + 6 new parity tests; the coverage-gate test was replaced, not added).

- [ ] **Step 5: Commit**

```bash
git add tests/test_client.py tests/test_coverage_gate.py
git commit -m "test: pin element_summaries semantics before the fetch goes concurrent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Phase 1 — cache index

**Files:**
- Modify: `fpl/data/cache.py:1-4` (imports), `:116-122` (`__init__`, `_paths`), `:124-137` (`put`), `:200-207` (`prune`)
- Test: `tests/test_cache.py` (append), `tests/test_perf_contracts.py` (remove one xfail)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Cache` keeps its public API (`put`, `newest`, `newest_stamp`, `newest_meta`, `get_fresh`, `prune`) and adds `Cache.refresh() -> None`. The index is built with `os.scandir(self.root)`, called through the `os` module attribute, which is what the scan-count contract patches.

- [ ] **Step 1: Write the failing index tests (append to `tests/test_cache.py`)**

```python
# --- the slug index ---------------------------------------------------------

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


def test_refresh_sees_another_writer(tmp_path):
    reader, writer = Cache(tmp_path), Cache(tmp_path)
    assert reader.newest("fixtures") is None
    writer.put("fixtures", [9], now=NOW)
    reader.refresh()
    assert reader.newest("fixtures")[0] == [9]
```

- [ ] **Step 2: Run them to see which fail**

Run: `python -m pytest tests/test_cache.py -v -p no:cacheprovider -k "index or refresh or prune_updates or put_after"`
Expected: `test_index_ignores_files_that_are_not_snapshots` FAILS (`fixtures_notes.json` sorts above the real snapshot, so `newest` raises `ValueError` from `strptime`), and `test_refresh_sees_another_writer` FAILS (`AttributeError: 'Cache' object has no attribute 'refresh'`). The other three pass on the glob code; they are regression guards for the index.

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
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # slug -> snapshot paths, newest first. Built on first use by ONE scan of
        # the directory and kept current by put/prune. Globbing per lookup
        # listed the whole cache for every call, and a weekly run makes ~2,700
        # of them against ~4,000 files: 22 of its 40 seconds.
        self._index: dict[str, list[Path]] | None = None

    def refresh(self) -> None:
        """Forget the index so the next lookup rescans the directory.

        For a caller that knows another process has written to the cache. One
        run owns the cache at a time today, so nothing calls this yet.
        """
        self._index = None

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

    def _build_index(self) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        with os.scandir(self.root) as entries:
            for entry in entries:
                slug = self._slug_of(entry.name)
                if slug is not None and entry.is_file():
                    index.setdefault(slug, []).append(self.root / entry.name)
        for paths in index.values():
            paths.sort(reverse=True)
        return index

    def _paths(self, slug: str) -> list[Path]:
        if self._index is None:
            self._index = self._build_index()
        return list(self._index.get(slug, ()))
```

In `put`, after the sidecar write and before `return p`, add:

```python
        if self._index is not None:
            paths = self._index.setdefault(slug, [])
            if p not in paths:
                paths.append(p)
                paths.sort(reverse=True)
        return p
```

Replace `prune` (lines 200–207) with:

```python
    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)
        removed = 0
        for p in paths[keep:]:
            p.unlink()
            p.with_suffix(META_SUFFIX).unlink(missing_ok=True)
            removed += 1
        if self._index is not None:
            self._index[slug] = paths[:keep]
        return removed
```

(The `put` docstring's line "`_paths` globs for snapshots and would otherwise try to read it as one" should now read "`_paths` indexes `*.json` snapshots and would otherwise try to read it as one.")

- [ ] **Step 4: Run the cache tests and the scan contract**

Remove the `@pytest.mark.xfail(...)` line above `test_cache_scans_directory_once` in `tests/test_perf_contracts.py`.

Run: `python -m pytest tests/test_cache.py tests/test_perf_contracts.py -v -p no:cacheprovider`
Expected: every cache test PASSES; `test_cache_scans_directory_once` PASSES; the three phase 2 contracts still xfail.

- [ ] **Step 5: Full suite, golden check, bench**

```bash
python -m pytest -q -p no:cacheprovider        # expect 786 passed, 3 xfailed
python scripts/bench.py golden check           # expect golden: OK
python scripts/bench.py run --label phase1-cache-index
```
Expected bench: `cache_hits_s` under 1 s, and `gw6_cache_only_s` ≤ 20 (success criterion 2). `refresh` is still about 650 s extrapolated; that's phase 2. Fill the `phase1-cache-index` column in `docs/perf/README.md`.

- [ ] **Step 6: Commit**

```bash
git add fpl/data/cache.py tests/test_cache.py tests/test_perf_contracts.py docs/perf
git commit -m "perf: index cache snapshots by slug instead of globbing per lookup

One directory scan per Cache instead of ~2,700 on a weekly run.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Phase 2a — `TokenBucket` rate limiter

**Files:**
- Create: `fpl/data/throttle.py`
- Test: `tests/test_throttle.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TokenBucket(rate: float, clock=time.monotonic, sleep=time.sleep)` with `acquire() -> None` (blocks until this caller may start a request) and `pause_for(seconds: float) -> None` (no acquisition by any thread before now + seconds). `rate <= 0` disables spacing, but `pause_for` still applies. Thread-safe. Task 6 uses exactly these names.

- [ ] **Step 1: Write the failing tests (`tests/test_throttle.py`)**

```python
import threading
import time

from fpl.data.throttle import TokenBucket


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


class TokenBucket:
    """At most `rate` acquisitions per second across all threads sharing it.

    Capacity is one: acquisitions are spaced 1/rate apart and an idle bucket
    banks nothing, so a pause never turns into a burst -- FPL's edge answers
    bursts with 429s. `rate <= 0` disables the spacing (tests, and anyone who
    explicitly opts out) but still honours `pause_for`, which is how a 429
    seen by one worker holds back all of them.
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

    def acquire(self) -> None:
        # Reserve a start time under the lock, sleep outside it, so one waiting
        # thread never blocks the others from reserving theirs.
        with self._lock:
            now = self._clock()
            start = max(now, self._next, self._paused_until)
            if self.rate > 0:
                self._next = start + 1.0 / self.rate
            wait = start - now
        if wait > 0:
            self._sleep(wait)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_throttle.py -v -p no:cacheprovider`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add fpl/data/throttle.py tests/test_throttle.py
git commit -m "feat: thread-safe token-bucket limiter for the FPL fetch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Phase 2b — concurrent `element_summaries`, retries, config, call sites

**Files:**
- Modify: `fpl/data/client.py` (whole module; `_get` keeps its behaviour)
- Modify: `fpl/config.py:44-45` (dataclass fields) and `:119-120` (loader), `config.yaml` (`data:` block)
- Modify: `run_gameweek.py:96` and `:109-111` (progress comment), `fpl/pipeline.py:577`, `scripts/run_backtest.py:176`, `scripts/score_gameweek.py:60`
- Test: `tests/test_client.py` (append), `tests/test_config.py` (append), `tests/test_perf_contracts.py` (remove three xfails)

**Interfaces:**
- Consumes: `TokenBucket(rate)`, `.acquire()`, `.pause_for(seconds)` from Task 5; the Task 3 parity tests as the behaviour contract.
- Produces: `FplClient(cache, ttl_hours=6, rate_limit_s=1.0, session=None, fetch_workers=4, fetch_rate_per_s=None)`. When `fetch_rate_per_s` is `None`, it resolves to `0.0` if `rate_limit_s` is falsy, else `DEFAULT_FETCH_RATE = 5.0`. The module constants are `RETRY_STATUSES`, `MAX_RETRIES = 3`, `BACKOFF_S = (1.0, 2.0, 4.0)` and `MAX_RETRY_AFTER_S = 60.0`; tests monkeypatch `BACKOFF_S`. `Config.fetch_workers: int = 4` and `Config.fetch_rate_per_s: float = 5.0`.

- [ ] **Step 1: Write the failing retry, interrupt and dedupe tests (append to `tests/test_client.py`)**

```python
# --- concurrent fetch: retries, interruption, duplicates ----------------------
import threading
import time

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

    def __init__(self, scripts):
        self.scripts = {u: list(s) for u, s in scripts.items()}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def get(self, url, timeout=None):
        with self.lock:
            self.calls.append(url)
            steps = self.scripts[url]
            step = steps.pop(0) if len(steps) > 1 else steps[0]
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


def test_an_interrupted_refresh_stops_promptly_and_keeps_what_it_fetched(tmp_path):
    class SlowSession:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def get(self, url, timeout=None):
            with self.lock:
                self.calls += 1
            time.sleep(0.1)
            return ScriptedResponse({"url": url})

    class Stop(Exception):
        pass

    def progress(done, total):
        if done == 3:
            raise Stop

    s = SlowSession()
    cache = Cache(tmp_path)
    c = FplClient(cache, rate_limit_s=0, session=s, fetch_workers=2, fetch_rate_per_s=0)

    t = time.monotonic()
    with pytest.raises(Stop):
        c.element_summaries(range(1, 41), progress=progress)
    elapsed = time.monotonic() - t

    # Draining the queue would take 40 x 0.1 / 2 = 2 s.
    assert elapsed < 1.0
    assert s.calls < 40
    assert sum(1 for p in range(1, 41) if cache.newest(f"element-summary-{p}")) >= 3


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
Expected: most Step 1 tests FAIL or ERROR (the `no_backoff` fixture can't patch the missing `BACKOFF_S`; `_limiter` and `fetch_workers` don't exist; the old code dedupes nothing and never retries); the Step 2 tests FAIL (`AttributeError: 'Config' object has no attribute 'fetch_workers'`). Every Task 3 parity test still PASSES.

- [ ] **Step 4: Rewrite `fpl/data/client.py`**

Keep the module docstring, `BASE`, `UA`, `FORBIDDEN`, `HISTORY_TTL_H`, `DataCoverageError`, `_record_source`, `source_summary`, `_throttle`, `bootstrap`, `fixtures`, `element_summary`, `entry`, `entry_history` and `entry_picks` exactly as they are. Change the imports and constants, `__init__`, `_get` (refactored onto shared helpers, same behaviour) and `element_summaries`, and add `_cached`, `_store`, `_fallback`, `_worker_session`, `_close_worker_sessions`, `_fetch_json` and `_retry_after`:

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .cache import Cache
from .throttle import TokenBucket

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; fpl-team-picker/1.0)"}
FORBIDDEN = ("my-team",)
# `history_past` for a completed season is immutable, so these can cache hard.
HISTORY_TTL_H = 24 * 30
# Element-summaries are the one bulk fetch: ~667 of them after every gameweek.
# One at a time behind a 1 s throttle that was 11+ minutes; a few in flight
# under a shared cap keeps the same total politeness budget per second while
# the network latency overlaps.
DEFAULT_FETCH_RATE = 5.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_S = 60.0
```

`__init__` (the signature changes; the body keeps every existing attribute and comment, then adds the fetch machinery):

```python
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0,
                 session=None, fetch_workers: int = 4,
                 fetch_rate_per_s: float | None = None):
        self.cache = cache
        self.ttl_hours = ttl_hours
        self.rate_limit_s = rate_limit_s
        self.session = session or requests.Session()
        # (Keep today's lines 33-51 verbatim here: stale, sources, snapshot_meta,
        #  unverified, fetch_failures and _last_call, with their comments.)
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
```

The shared helpers and the refactored `_get`:

```python
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
```

The worker-side HTTP:

```python
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
    def _retry_after(resp) -> float:
        value = (getattr(resp, "headers", None) or {}).get("Retry-After")
        try:
            return min(max(float(value), 0.0), MAX_RETRY_AFTER_S)
        except (TypeError, ValueError):
            return 0.0

    def _fetch_json(self, url: str):
        """GET `url` on a worker thread, through the shared limiter.

        Retries what is worth retrying -- rate limiting, server errors, dropped
        connections -- with backoff; a 404 or anything unexpected fails at
        once. Touches no cache and no client state: the calling thread does
        all of that, so none of it needs a lock.
        """
        if any(f in url.lower() for f in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        session = self._worker_session()
        for attempt in range(MAX_RETRIES + 1):
            last = attempt == MAX_RETRIES
            self._limiter.acquire()
            try:
                resp = session.get(url, timeout=30)
            except (requests.ConnectionError, requests.Timeout):
                if last:
                    raise
                time.sleep(BACKOFF_S[attempt])
                continue
            status = getattr(resp, "status_code", 200)
            if status in RETRY_STATUSES and not last:
                wait = max(BACKOFF_S[attempt], self._retry_after(resp))
                if status == 429:
                    # FPL is telling us all to slow down, not just this worker.
                    self._limiter.pause_for(wait)
                else:
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
```

`element_summaries` (keep the existing docstring and add the paragraph shown at its end):

```python
    def element_summaries(self, player_ids, ttl_hours: float = HISTORY_TTL_H,
                          progress=None, not_before=None,
                          require_final_through=None) -> dict[int, dict]:
        """...existing docstring, unchanged, then:

        Cache hits are resolved first, on this thread. Only the misses go to
        the network, `fetch_workers` at a time under the shared rate cap, and
        every result is written back on this thread as it arrives -- so the
        cache and the bookkeeping above never see two threads at once.
        """
        ids = list(dict.fromkeys(int(pid) for pid in player_ids))
        found: dict[int, dict] = {}
        done = 0

        def tick() -> None:
            nonlocal done
            done += 1
            if progress:
                progress(done, len(ids))

        misses = []
        for pid in ids:
            cached = self._cached(f"element-summary-{pid}", ttl_hours, not_before,
                                  require_final_through)
            if cached is None:
                misses.append(pid)
            else:
                found[pid] = cached
                tick()

        if misses:
            pool = ThreadPoolExecutor(max_workers=self.fetch_workers)
            try:
                futures = {pool.submit(self._fetch_json, f"{BASE}element-summary/{pid}/"): pid
                           for pid in misses}
                for fut in as_completed(futures):
                    pid = futures[fut]
                    slug = f"element-summary-{pid}"
                    try:
                        payload = fut.result()
                    except Exception:
                        payload = self._fallback(slug)
                        if payload is None:
                            self.stale = True
                            self.fetch_failures.add(pid)
                    else:
                        self._store(slug, payload)
                    if payload is not None:
                        found[pid] = payload
                    tick()
            except BaseException:
                # Ctrl-C, or a progress callback raising: stop now rather than
                # drain hundreds of queued requests. Whatever already arrived
                # is on disk, so a re-run resumes from the cache.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
            finally:
                self._close_worker_sessions()

        return {pid: found[pid] for pid in ids if pid in found}
```

- [ ] **Step 5: Add the config keys**

`fpl/config.py`, after `cache_ttl_matchday_hours: int = 1` (line 45):

```python
    # The bulk element-summary fetch after each gameweek: requests in flight
    # and the global cap across them. 1 worker at 1.0/s is the old sequential
    # crawl -- the lever to pull if FPL starts refusing us.
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

Each of these four lines gains `fetch_workers=cfg.fetch_workers, fetch_rate_per_s=cfg.fetch_rate_per_s`:

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

In `run_gameweek.py`, the progress comment at line 110 changes from "Player history is fetched one request per second on a cold cache, so a first run takes minutes" to "Player history is fetched a few requests at a time on a cold cache, so a first run still takes a couple of minutes".

Verify that nothing constructs a production client without the settings:
Run: `grep -rn "FplClient(" fpl scripts run_gameweek.py`
Expected: the four lines above, plus the two in `scripts/bench.py` (those intentionally use FplClient's own defaults, so the bench measures what a bare client does).

- [ ] **Step 7: Remove the three phase 2 xfail markers and run the targeted tests**

Delete the `@pytest.mark.xfail(...)` lines above `test_refresh_is_concurrent`, `test_refresh_respects_worker_bound` and `test_refresh_respects_rate_cap` in `tests/test_perf_contracts.py`.

Run: `python -m pytest tests/test_client.py tests/test_config.py tests/test_perf_contracts.py tests/test_coverage_gate.py tests/test_throttle.py -v -p no:cacheprovider`
Expected: all PASS, including every Task 3 parity test unchanged.

- [ ] **Step 8: Full suite, golden check, bench**

```bash
python -m pytest -q -p no:cacheprovider        # expect 805 passed, 0 xfailed
python scripts/bench.py golden check           # expect golden: OK
python scripts/bench.py run --label phase2-concurrent-fetch
```
Expected bench: `refresh.req_per_s` ≈ 5 and `refresh.extrapolated_667_s` ≈ 135–150 (success criterion 1: ≤ 180). `cache_hits_s` and `gw6_cache_only_s` should be about the same as phase 1. Fill the `phase2-concurrent-fetch` column in `docs/perf/README.md`.

- [ ] **Step 9: Commit**

```bash
git add fpl/data/client.py fpl/config.py config.yaml run_gameweek.py fpl/pipeline.py scripts/run_backtest.py scripts/score_gameweek.py tests/test_client.py tests/test_config.py tests/test_perf_contracts.py docs/perf
git commit -m "perf: fetch element-summaries concurrently under a shared rate cap

4 in flight at 5 req/s with 429/5xx backoff: a post-gameweek refresh drops
from 11+ minutes to about 2.5. Cache writes and failure bookkeeping stay on
the calling thread; fetch_workers: 1 / fetch_rate_per_s: 1.0 restores the
old sequential crawl.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Live smoke test (manual; needs the network; the only step that talks to FPL)**

In a scratch copy of the repo (never the worktree's own `data/`), delete ~50 element-summary snapshots so they must be refetched, then run the real pipeline once:

```bash
S=$(mktemp -d) && cp -r fpl run_gameweek.py config.yaml data "$S"/ && cd "$S" \
  && ls data/cache/element-summary-*.json | head -150 | xargs rm \
  && time python run_gameweek.py --mode 2 --gw 6 | tail -5
```
Expected: "Fetching player history" completes ~50 refetches in ≈ 10–15 s, no 429 storms (a stray retry is fine), and a report renders without errors. It will **not** necessarily match the golden decision: without `--no-refresh`, bootstrap and fixtures are refreshed live too. If FPL refuses requests, lower `fetch_rate_per_s` in `config.yaml` and report it. Don't raise the default.

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
python -m pytest -q -p no:cacheprovider        # expect 806 passed
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
