# Fetch and cache performance — design

Date: 2026-09-22
Status: approved (2026-09-22)
Branch / worktree: `worktree-perf-fetch-and-cache` at `.claude/worktrees/perf-fetch-and-cache`

## Problem

The weekly run is slow in two separate ways.

**Refresh runs.** The first run after a gameweek finishes has to refetch every
player's element-summary: `pipeline.run` passes
`not_before=data_complete_after(fixtures)`, so the finished gameweek makes every
cached summary too old to use. `FplClient.element_summaries` fetches the ~667
players one at a time, and `_throttle` holds request starts at least
`rate_limit_s = 1.0` s apart. So the floor for a refresh is 667 s, about 11 minutes,
plus network latency. The cache confirms it: all 667 summaries were rewritten
at 2026-09-21 08:xx, the morning after GW5.

**Cache-only runs** (`--no-refresh`, or any later run the same week). Profiled
on 2026-09-22 against a scratch copy of `data/`, Mode 2, GW6: **40.6 s** wall
time.

| Hotspot | Time | Share | Cause |
|---|---|---|---|
| `Cache._paths` (`fpl/data/cache.py:121`) | 22.2 s | 55% | 2,674 `glob` calls, each scanning all 4,061 cache files: 10.7M regex matches. **O(lookups × files).** |
| `build_xp` (`fpl/model/xp.py:324`) | 7.8 s | 19% | Nested `iterrows` over players × fixtures. The fixture frame is boolean-filtered once per player, a Series is copied per fixture, and scalar `poisson.sf` is called ~31k times. |
| Transfer MILP (`transfers._solve` ×5) | 3.2 s | 8% | External CBC; mostly waiting on the subprocess. Out of scope. |
| `blended_rates` | 2.1 s | 5% | Row-wise pandas. Out of scope. |

Each cache lookup triggers up to four `_paths` scans (`get_fresh` → `newest`
and `newest_meta`, `_record_source` → `newest_stamp`, and `newest_meta` again
for the `unverified` check). A fetch adds `put` and `prune`. The glob cost
therefore also taxes every request during a refresh.

## Goals and success criteria

1. **Refresh run** at default settings: 11+ min → **≤ 3 min**.
2. **Cache-only run** (GW6, Mode 2): 40.6 s → **≤ 20 s** after the cache index,
   **≤ 16 s** if the optional xP phase lands. (An earlier draft said 12 s. The
   phase 3 changes remove about 3 s of the 7.8 s in `build_xp`. The rest is
   row-wise pandas that phase 3 deliberately leaves alone.)
3. **Same outputs, with three intentional fetch changes.** (Revised after two
   plan reviews: the first draft claimed "no behaviour change", which retries,
   de-duplication and the long-Retry-After stop make false.)
   - Whenever every fetch succeeds or fails permanently (a 4xx other than 429,
     retries exhausted, or no network), `element_summaries` returns the same
     dict and leaves `fetch_failures`, `stale`, `unverified` and `sources`
     exactly as the current code does. That includes per-player containment:
     a corrupt snapshot, or a failed cache write, prune or fallback read, fails
     that player only and the fetch carries on. A corrupt fresh snapshot is
     not refetched, and a failed write is not rescued by a fallback, as today.
   - Intentional difference 1: transient failures (429, 5xx, connection errors,
     timeouts) are retried up to 3 times before counting as failures, so a blip
     that used to produce a stale fallback now produces fresh data.
   - Intentional difference 2: duplicate ids are fetched once, and `progress`
     totals count unique ids. No caller passes duplicates today.
   - Intentional difference 3: a `Retry-After` longer than 120 s stops the
     fetch for the rest of the run, and the players not yet fetched take the
     stale-fallback / `fetch_failures` path. Today every later player would
     still be attempted, against the server's explicit request.
   - The GW6 golden run gives the same squad, starting XI, bench order,
     captain, vice-captain, transfers and chip advice, with the clock frozen
     at the capture instant.
   - Every numeric column of the xP frame matches within **1e-4 absolute**.
     `build_xp` rounds to 4 dp, so this allows one rounding tick.
4. All 774 existing tests pass after every phase.

## Non-goals

- Incremental refresh via `event/{gw}/live/` (the "B" option). It would cut 667
  requests to a handful, but it rebuilds the history rows the minutes and form
  models read, and it interacts with the `final_through` / `not_before`
  safeguards. Revisit only if phase 2 is not enough.
- Skipping players at fetch time (it changes what the coverage gate means).
- The MILP solver, `blended_rates`, and test-suite speed as a target in itself.
- Making the non-summary endpoints (bootstrap, fixtures, entry) concurrent.
  There are only a few of them per run.

## Design

### Phase 0 — Baseline (no runtime changes)

**`scripts/bench.py`** writes one JSON file per invocation to
`docs/perf/<label>.json`. It records the commit, Python version, machine and
per-benchmark wall time (median of 3). Three benchmarks:

- **`cache_hits`**: `FplClient.element_summaries` over a synthetic cache in a
  temp dir: 667 slugs × 3 snapshots each, with `.meta` sidecars, plus ~2,000
  unrelated files so the directory is realistically large. Every lookup is a
  cache hit. This isolates the `_paths` cost.
- **`refresh`**: `element_summaries` over N players (default 30) with an empty
  cache and a fake session that sleeps a fixed 150 ms per request. It uses the
  fetch settings from `config.yaml`, as `run_gameweek` does. Before phase 2
  there are none, and the client's 1 s throttle is the production path. It
  reports wall time, requests/s and the extrapolation to 667 players.
- **`gw6_cache_only`**: `run_gameweek.py --mode 2 --gw 6 --no-refresh` as a
  subprocess, against a scratch copy of `data/` (a plan-only run still writes a
  forecast version, and the real manifest must not collect benchmark entries).
  Skipped with a clear message if `data/cache` is absent (mobile/cloud
  checkouts).

**Golden output.** `scripts/bench.py golden capture` runs the GW6 cache-only
pipeline once on unmodified master. It writes `docs/perf/golden-gw6/`:
`decision.json` (squad, XI, bench order, captain, vice-captain, transfers,
chip), `xp.parquet` and `clock.txt`. `golden check` re-runs the pipeline and
diffs against these files with the tolerance above.

The pipeline reads the wall clock (override ages in `fpl/data/overrides.py`,
the matchday check in `fpl/pipeline.py`), so both runs execute with
`datetime.now()` frozen at the instant in `clock.txt`. The frozen class needs a
metaclass so that `isinstance(real_datetime, datetime)` stays true in the
patched modules. The golden run uses the real `data/cache`, so it is a local
check, not a CI test.

**`tests/test_perf_contracts.py`** holds deterministic, CI-safe tests that
assert *complexity, not wall time*:

- `test_cache_scans_directory_once`: counts directory scans (patch
  `Path.glob` / `os.scandir` on the cache root) across 667 lookups on the
  synthetic cache and asserts it is ≤ 1.
- `test_refresh_respects_worker_bound`: the fake session records the peak
  number of concurrent in-flight requests; assert ≤ `fetch_workers`.
- `test_refresh_respects_rate_cap`: the fake session records request start
  times; assert that no 1-second window holds more than `fetch_rate_per_s`
  starts (plus one for bucket burst).
- `test_refresh_is_concurrent`: 40 requests × 150 ms at workers=4 and a high
  rate cap finish well under the 6 s sequential floor.

Each of these fails on current code and is committed marked
`xfail(strict=True)`. The phase that satisfies a test removes its marker.

**Parity tests for fetch semantics**, added to `tests/test_client.py` before
phase 2. They pin the current sequential behaviour of `element_summaries` for:
all hits; all misses; mixed; one player's request raising with a cached
fallback (goes to `stale`, still returned); one raising with no cache (goes to
`fetch_failures`, omitted); a pre-marker snapshot with `require_final_through`
(goes to `unverified`); and progress-callback calls (count reaches `len(ids)`,
`total` is constant). Containment is pinned too: a corrupt fresh snapshot, a
failing `cache.put` and a corrupt fallback snapshot each fail only their own
player, and a raising progress callback still stops the fetch. They must pass
on both the old and the new code (verified against today's client: 28/28).

`tests/test_coverage_gate.py::test_the_client_records_which_players_failed`
calls `FplClient.element_summaries` unbound, on an object built with
`__new__` whose `element_summary` is overridden. It tests an implementation
detail that phase 2 removes. Phase 0 rewrites it to drive a real `FplClient`
through a session that fails for player 3. It keeps its assertion
(`fetch_failures == {3}`) and passes on both the old and the new code.

### Phase 1 — Cache index (`fpl/data/cache.py`)

- `Cache` gets a private `_index: dict[str, list[Path]] | None`, built lazily
  on first use by a single `os.scandir` of `root`. A filename belongs to slug
  `s` when it matches `f"{s}_*.json"`. Parse it by splitting the stem on the
  last `_` and checking that the suffix parses with `TS_FMT`. `.meta` sidecars
  are excluded, as they are today.
- Each slug's list is kept sorted newest-first, the same ordering
  `sorted(glob, reverse=True)` gives today (lexicographic on the name, which
  the fixed-width `TS_FMT` makes chronological).
- `put` inserts the new path into the slug's list in position. `prune` removes
  the unlinked paths. `_paths(slug)` returns a copy of the list, or `[]`.
- **Other writers.** One writer per directory is the working assumption: only
  one `Cache` per process touches `data/cache`. A concurrent second run is
  possible, though, because the script runs headless. So the index remembers
  the directory's `st_mtime_ns`, and every lookup stats the directory and
  rebuilds when it has changed.
  - `put` and `prune` fold their own writes in and adopt the new mtime, so they
    never trigger a rescan. If someone else wrote first, the index is dropped
    instead.
  - Timestamps are tick-granular, so this detection is best-effort. The
    guarantee is narrower: a snapshot that vanished (pruned elsewhere) makes
    `newest` rescan once instead of crashing, and `prune` uses
    `unlink(missing_ok=True)`.
- New `refresh()` public method drops the index outright.
- The public API and return types do not change.

Complexity: one O(F) scan per `Cache` instance (plus one per external change),
then O(1) per lookup plus one `stat`, and O(k) per `put`/`prune` for a slug with
k snapshots (k ≤ 3 after pruning).

### Phase 2 — Concurrent element-summary fetch (`fpl/data/client.py`)

Only `element_summaries` changes. `_get` and every single-endpoint method keep
their current sequential behaviour and the `rate_limit_s` throttle.

Three passes:

1. **Resolve from cache (main thread).** For each id in input order, run the
   same cache check `_get` does today (`get_fresh` with the given TTL,
   `not_before` and `require_final_through`). On a hit, record the source and
   the `unverified` flag, store the payload, and tick progress. Collect the
   misses.
2. **Fetch misses (worker threads).** Run a `ThreadPoolExecutor(max_workers=W)`
   whose task does HTTP only: check the forbidden-endpoint list, take a token
   from the shared limiter, call `session.get(url, timeout=30)`, then
   `raise_for_status()` and `.json()`. Each worker thread gets its own
   `requests.Session` via `threading.local()`. When the client was built with an
   injected `session` (the tests do this), every worker uses that one object
   instead.
   - **Rate limiter:** a token bucket with capacity 1 and refill rate
     R tokens/s, guarded by a `threading.Lock`. It **reserves nothing**: each
     wake-up re-reads the next free slot and the shared pause under the lock.
     So a pause set after a 429 also holds back workers that were already
     asleep waiting for their turn. (The first draft booked future slots, and
     those workers would have started inside the pause.) R = 0 disables the
     spacing but not pauses or cancellation.
   - **Retries:** on HTTP 429, or on a 5xx, connection error or timeout, retry
     up to 3 times with exponential backoff (1 s, 2 s, 4 s), or the server's
     `Retry-After` when it is larger. `Retry-After` is parsed as delay-seconds
     or an HTTP-date (RFC 9110) and is **never shortened**. A 429 sets the
     limiter's shared pause, so every worker backs off together. A 404, other
     4xx errors and unexpected exceptions are not retried.
   - **A wait longer than a run can sit out** (`Retry-After` > 120 s) is obeyed
     by not asking again this run. The fetch stops, and every player not yet
     fetched takes the stale-fallback / `fetch_failures` path, the same as in
     an outage. The coverage gate then decides whether the run may proceed.
   - **Bounded and cancellable.** At most `2 × W` futures are queued at once.
     One `threading.Event` is checked before each rate wait, request, backoff
     and retry, and the rate and backoff waits are `Event.wait`, so they wake
     immediately when it is set.
3. **Integrate results (main thread).** Wait in 0.25 s slices
   (`wait(..., FIRST_COMPLETED)`); a blocking wait on Windows does not see
   Ctrl-C until it returns. On success: `cache.put(slug, payload,
   meta=snapshot_meta or None)`, `cache.prune(slug, keep=3)`,
   `_record_source(slug)`. On final failure: do the same as today, so use
   `cache.newest(slug)` as the stale fallback and set `stale`, or on no cache
   add the id to `fetch_failures` and omit it.
   - The returned dict is rebuilt in input-id order, so iteration order matches
     the old code.

All cache and client-state mutation stays on the main thread. The index from
phase 1 therefore needs no lock.

**Config** (`config.yaml` → `data:`, read in `fpl/config.py`):

```yaml
data:
  fetch_workers: 4        # element-summary requests in flight
  fetch_rate_per_s: 5.0   # global cap across workers; 0 = uncapped
```

`FplClient.__init__` gains `fetch_workers: int = 4` and
`fetch_rate_per_s: float | None = None`. `None` resolves to `0.0` when
`rate_limit_s == 0`, and to `5.0` otherwise. Existing tests construct
`FplClient(..., rate_limit_s=0)` and so stay unthrottled with no edits, while an
explicit value always wins. `Config` gains `fetch_workers` and
`fetch_rate_per_s`, read from the `data:` block in `fpl/config.py` next to
`cache_ttl_hours`. The production call sites (`run_gameweek.py`, the default
client in `pipeline.run`, `scripts/run_backtest.py` and
`scripts/score_gameweek.py`) pass both values.

Expected: 667 misses / 5 req/s ≈ 133 s, plus the latency tail, so about 2.2–2.5 min.

### Phase 3 — xP inner loop (optional, `fpl/model/xp.py`)

Only lands if phases 1–2 are merged and the golden check still passes.

- `expected_thresholds`: replace the Python loop of `MAX_THRESHOLDS` scalar
  `poisson.sf` calls with one call on
  `np.arange(1, MAX_THRESHOLDS + 1) * per_point - 1`, then `.sum()`.
- `build_xp`: group `tfx` by `team_id` once (`dict(tuple(tfx.groupby(...)))`)
  instead of boolean-filtering per player. Stop copying `fx` per fixture: pass
  `assist_feasibility_scale` to `xp_for_fixture` as an argument, with default
  1.0, so the function's other callers are unaffected.
- The formula does not change. Parity: the golden xP frame within 1e-4, plus
  the existing `tests/test_xp.py`.

## Error handling

- The per-player failure semantics are unchanged (see the phase 0 parity
  tests), apart from the three intentional differences in success criterion 3.
  A retried request that finally succeeds counts as a success.
- **Stopping.** On any exception in the calling thread (Ctrl-C, or a raising
  progress callback), the fetch:
  1. sets the cancel event;
  2. joins the pool (`shutdown(wait=True, cancel_futures=True)`). Every worker
     exits at its next check, but a request already on the wire cannot be
     interrupted, so this takes as long as the slowest one in flight: normally
     a fraction of a second, bounded by the 30 s request timeout;
  3. only then closes the per-thread sessions;
  4. re-raises.

  No request starts after `element_summaries` has left, no session is closed
  under a running request, and Python's exit-time thread join has nothing left
  to wait for. Summaries already integrated are on disk, so a re-run resumes
  from the cache: the same resumability the sequential loop has.
- **Rollback.** `fetch_workers: 1` and `fetch_rate_per_s: 1.0` restore today's
  request pacing: one request in flight, starts at least 1 s apart. Retries
  and de-duplication remain; they aren't separately switchable, because
  neither can make a run slower than a transient failure followed by a
  re-run. That is the lever if FPL starts refusing us.

## Testing summary

| Layer | What | Where | When |
|---|---|---|---|
| Complexity contracts | scans ≤ 1, in-flight ≤ W, rate ≤ R, concurrency real | `tests/test_perf_contracts.py` | CI, every phase |
| Fetch parity | hit/miss/fail/stale/unverified/progress semantics | `tests/test_client.py` | written before phase 2; passes before and after |
| Cache parity | ordering, put/prune/newest/newest_meta behaviour with the index | `tests/test_cache.py` | before and after phase 1 |
| Limiter | spacing, no burst, pause holds already-waiting threads, cancellation | `tests/test_throttle.py` | phase 2a |
| Fetch robustness | retries, Retry-After (seconds/date/too long), 404, interrupt stops workers, backoff cut short, dedupe | `tests/test_client.py` | phase 2b |
| Golden run | GW6 decisions identical; xP within 1e-4; clock frozen | `scripts/bench.py golden check` | locally, end of every phase |
| Benchmarks | wall time for cache_hits / refresh / gw6_cache_only | `scripts/bench.py` → `docs/perf/*.json` | baseline, then after every phase |
| Regression | the full existing suite (774) | `pytest` | every commit |

## Delivery

- Work in the git worktree `.claude/worktrees/perf-fetch-and-cache` (branch
  `worktree-perf-fetch-and-cache`), branched from `master` at `c3a5c8c`. The
  worktree has no `data/cache` (it is gitignored). Task 1 of the plan copies the
  main checkout's cache in once, which also freezes the golden run's inputs.
- Order: phase 0, then 1, 2, and optionally 3. Each phase is test-first and ends
  with the full suite passing, a golden check, a bench run and one commit
  (`docs/perf/` updated in the same commit).
- Done when phases 0–2 are committed with success criteria 1–4 met and the
  before/after numbers recorded in `docs/perf/`. Phase 3 is then a separate go
  or no-go decision.
