# Performance log — fetch and cache

Normal recommendation runs now require a live response for fixtures,
bootstrap-static, and every player in the live bootstrap list. Cached snapshots
are still written for provenance, but are not read or used as a fallback in a
normal run. An incomplete live refresh stops before a prediction is saved.
`--no-refresh` explicitly retains the older cache-first/fallback behavior and
must not be used when a fully current prediction is required. The timings below
are historical measurements of the cache-enabled and simulated-refresh paths,
not a live end-to-end timing for the new default.

Spec: `docs/superpowers/specs/2026-09-22-fetch-and-cache-perf-design.md`
Plan: `docs/superpowers/plans/2026-09-22-fetch-and-cache-perf.md`

Produced by `python scripts/bench.py run --label <phase>`; the raw JSON sits
next to this file. The golden check is `python scripts/bench.py golden check`.

| Benchmark | baseline | phase1-cache-index | phase2-concurrent-fetch | phase3-xp (optional) |
|---|---|---|---|---|
| cache_hits (667 lookups, synthetic cache), s | 29.11 | 1.786 | 1.607 | 1.597 |
| refresh, extrapolated to 667 players, s | 648.4 | 648.4 | 134.5 | 134.8 |
| refresh, requests/s | 1.03 | 1.03 | 4.96 | 4.95 |
| GW6 Mode-2 `--no-refresh`, s | 34.26 | 14.52 | 14.79 | 12.00 |
| Golden check | OK (captured) | OK | OK | OK |
