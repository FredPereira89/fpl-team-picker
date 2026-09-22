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


def test_cache_scans_directory_once(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    client = FplClient(Cache(tmp_path), rate_limit_s=0, session=NoNetwork())
    scans = _count_scans(monkeypatch, tmp_path)

    out = client.element_summaries(ids, require_final_through=5)

    assert len(out) == N
    # Globbing per lookup listed the whole directory ~3 times per player.
    assert scans["n"] <= 1


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


def test_refresh_respects_worker_bound(tmp_path):
    s = RecordingSession(latency_s=0.05)
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s,
                  fetch_workers=3, fetch_rate_per_s=0)

    c.element_summaries(range(1, 31))

    assert len(s.calls) == 30
    assert 2 <= s.peak <= 3


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
