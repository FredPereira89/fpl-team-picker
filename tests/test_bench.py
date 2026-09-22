"""Integrity checks for the performance harness itself."""

import json
import subprocess
import sys

import pandas as pd

from scripts import bench


def test_run_uses_the_median_of_three_refresh_samples(tmp_path, monkeypatch):
    samples = iter([
        {"seconds": 9.0, "req_per_s": 3.33, "extrapolated_667_s": 200.1},
        {"seconds": 6.0, "req_per_s": 5.0, "extrapolated_667_s": 133.4},
        {"seconds": 7.0, "req_per_s": 4.29, "extrapolated_667_s": 155.6},
    ])
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "PERF", tmp_path)
    monkeypatch.setattr(bench, "bench_cache_hits", lambda: 1.0)
    monkeypatch.setattr(bench, "bench_refresh", lambda: next(samples))
    monkeypatch.setattr(bench, "bench_gw6_cache_only", lambda: 2.0)

    assert bench.cmd_run("test-median") == 0
    result = json.loads((tmp_path / "test-median.json").read_text())
    assert result["refresh"]["seconds"] == 7.0
    assert result["refresh"]["req_per_s"] == 4.29


def test_offline_guard_stops_a_requests_call_before_network():
    script = bench.OFFLINE_GUARD + "\nimport requests\nrequests.get('https://example.invalid/')\n"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "offline benchmark attempted network" in proc.stderr


def test_compare_xp_accepts_one_rounding_tick_but_not_more():
    old = pd.DataFrame({"player_id": [1], "xp_next1": [0.0003]})
    one_tick = pd.DataFrame({"player_id": [1], "xp_next1": [0.0004]})
    beyond = pd.DataFrame({"player_id": [1], "xp_next1": [0.00041]})

    assert bench.compare_xp(old, one_tick) == []
    assert bench.compare_xp(old, beyond)
