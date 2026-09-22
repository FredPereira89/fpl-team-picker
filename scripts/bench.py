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

# A cache-only benchmark must fail on any network attempt, including a miss
# that the client would otherwise catch and replace with stale data. Raising
# BaseException (rather than Exception) keeps the guard outside the client's
# per-player failure containment. This code runs only in scratch subprocesses.
OFFLINE_GUARD = r'''
import requests

class OfflineNetworkAttempt(BaseException):
    pass

def _deny_network(self, method, url, *args, **kwargs):
    raise OfflineNetworkAttempt(f"offline benchmark attempted network: {url}")

requests.sessions.Session.request = _deny_network
'''

OFFLINE_RUNNER = OFFLINE_GUARD + r'''
import sys
import run_gameweek
raise SystemExit(run_gameweek.main(sys.argv[1:]))
'''

# Runs inside the temporary repo copy. It wraps `run` so the Recommendation and
# the xP frame can be captured without teaching run_gameweek a new flag, and it
# freezes the clock at the instant recorded when the golden output was
# captured: override ages (fpl/data/overrides.py) and the matchday check
# (fpl/pipeline.py) read datetime.now(), so without this a golden check a few
# days later could drift with no code change at all.
DRIVER = OFFLINE_GUARD + r'''
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
        (repo / "offline_run.py").write_text(OFFLINE_RUNNER)
        t = time.perf_counter()
        subprocess.run([sys.executable, "offline_run.py", "--mode", "2", "--gw", str(GW),
                        "--no-refresh"], cwd=repo, check=True, capture_output=True)
        return time.perf_counter() - t


def _median(fn, repeats: int) -> float:
    return statistics.median(fn() for _ in range(repeats))


def cmd_run(label: str) -> int:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    refresh = sorted((bench_refresh() for _ in range(3)),
                     key=lambda sample: sample["seconds"])[1]
    result = {
        "label": label, "commit": commit, "python": platform.python_version(),
        "machine": platform.platform(), "when": datetime.now(timezone.utc).isoformat(),
        "cache_hits_s": round(_median(bench_cache_hits, 3), 3),
        "refresh": refresh,
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
