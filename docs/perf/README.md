# Performance log — fetch and cache

Spec: `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`
Plan: `docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md`

Produced by `python scripts/bench.py run --label <phase>`; the raw JSON sits
next to this file. The golden check is `python scripts/bench.py golden check`.

| Benchmark | baseline | phase1-cache-index | phase2-concurrent-fetch | phase3-xp (optional) |
|---|---|---|---|---|
| cache_hits (667 lookups, synthetic cache), s | 29.11 | 1.786 | 1.678 | — |
| refresh, extrapolated to 667 players, s | 648.4 | 648.4 | 135.8 | — |
| refresh, requests/s | 1.03 | 1.03 | 4.91 | — |
| GW6 Mode-2 `--no-refresh`, s | 34.26 | 14.52 | 15.00 | — |
| Golden check | OK (captured) | OK | OK | — |
