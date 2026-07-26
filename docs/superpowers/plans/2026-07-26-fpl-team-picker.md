# FPL Team Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only Python decision-support system that recommends an FPL squad, starting XI, captain, and transfers from data, producing a GW1 recommendation before the 2026/27 deadline (Fri 21 Aug 2026 18:30 UK).

**Architecture:** A linear pipeline — data layer (HTTP + cache + normalize) → xP model → backtest gate → MILP optimizer → report. Two swap seams keep it modular: `model/xp.py` emits a fixed-column DataFrame that `optimize/` is the sole consumer of, and `optimize/` emits a `Recommendation` that `report/` only formats, never recomputes.

**Tech Stack:** Python 3.13.1, pandas 2.2.1, numpy 1.26.4, scipy 1.14.1, pulp 2.9.0, pyarrow 18.1.0, requests 2.31.0, pytest, PyYAML.

**Spec:** `docs/superpowers/specs/2026-07-25-fpl-team-picker-design.md` — authoritative. Read it before starting.

## Global Constraints

- **Read-only, always.** No POST/PUT/DELETE. Never call `my-team/`, `get_my_team`, or `check_fpl_authentication`. No endpoint requiring a login/session cookie.
- **Budget ceiling £100.0m**; squad exactly 15 = 2 GK / 5 DEF / 5 MID / 3 FWD; **max 3 players per real club**; valid XI = 1 GK + 3–5 DEF + 2–5 MID + 1–3 FWD = 11.
- **Rate limit ≤ ~1 request/second.** `element-summary` calls sequential with delay.
- **Cache freshness:** reuse snapshot < 6h old, < 1h on matchdays. "Matchday" = any UTC date with ≥1 fixture `kickoff_time` in the cached fixture list.
- **Cache retention:** raw JSON at `data/cache/{slug}_{ISO8601}.json`, keep newest 3 per slug.
- **`now_cost` is tenths of a million** — divide by 10 on ingest. Resolve team and `element_type` IDs to names before anything downstream sees them.
- **Never fabricate a number.** Missing data is reported as missing. Uncertainty is stated explicitly.
- **Report phrasing is recommendation-only** — "recommend"/"suggest", never "transferred"/"captained".
- **Free transfers: `0 ≤ FT ≤ 5`**, survive Wildcard/Free Hit, accrue +1/GW to the cap.
- **FPL scoring constants** (used across model tasks — copy exactly):
  - Appearance: 1 pt for any minutes, 2 pts for 60+
  - Goal: GK 6, DEF 6, MID 5, FWD 4 | Assist: 3 (all positions)
  - Clean sheet: GK 4, DEF 4, MID 1, FWD 0 | Goals conceded: GK/DEF −1 per 2
  - Saves: GK +1 per 3 | Penalty save: +5 | Penalty miss: −2
  - Yellow −1, Red −3, Own goal −2
  - **Defensive Contribution (direct points, 2025/26 rule):** DEF 2 pts at ≥10 CBIT (clearances+blocks+interceptions+tackles); MID/FWD 2 pts at ≥12 CBIRT (adds recoveries). Per match, so model as `p(actions ≥ threshold)` from a per-90 rate — never compare a season total to the threshold.
  - The same raw actions **also** feed BPS. DC component owns the threshold award; `bps.py` owns bonus only. Additive, not double-counted.

---

## File Structure

| File | Responsibility |
|---|---|
| `config.yaml` | User preferences |
| `fpl/config.py` | Load + validate config into a `Config` dataclass |
| `fpl/state.py` | Persisted FT balance + chip usage → `data/state.json` |
| `fpl/data/cache.py` | Snapshot write/read, freshness, retention |
| `fpl/data/client.py` | HTTP against the 6 public endpoints, rate-limited, cache-aware |
| `fpl/data/normalize.py` | Raw JSON → tidy DataFrames, ID resolution, price/status conversion |
| `fpl/data/store.py` | Parquet persistence |
| `fpl/data/archive.py` | vaastav per-GW CSV loader (backtest only) |
| `fpl/model/strength.py` | Derived team attack/defence ratings, optional odds provider |
| `fpl/model/minutes.py` | `p_start`, `p_60`, `e_minutes`, availability flags |
| `fpl/model/scoring.py` | Per-90 rates with shrinkage, form blend |
| `fpl/model/fixtures.py` | Per-fixture difficulty, `p_CS`, DGW/BGW detection |
| `fpl/model/bps.py` | Expected bonus points |
| `fpl/model/xp.py` | Assembles all components → the contract DataFrame |
| `fpl/backtest/aggregate.py` | Tier 1: multi-season `history_past` validation |
| `fpl/backtest/gw_level.py` | Tier 2: per-GW backtest, metrics, trust gate |
| `fpl/optimize/squad.py` | Mode 1 MILP |
| `fpl/optimize/lineup.py` | XI / formation / bench order / captain |
| `fpl/optimize/transfers.py` | Mode 2 MILP |
| `fpl/optimize/chips.py` | Chip heuristics |
| `fpl/report/weekly.py` | Renders the `weekly-report` format |
| `run_gameweek.py` | CLI entry point, pure Python, no MCP |

---

### Task 0: Project scaffold, git, config

**Files:**
- Create: `requirements.txt`, `.gitignore`, `config.yaml`, `fpl/__init__.py`, `fpl/config.py`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Config` dataclass with attributes `budget: float`, `horizon_gw: int`, `risk_profile: str`, `ownership_weight: float`, `news_weight: float`, `news_max_age_hours: int`, `form_half_life_gw: float`, `form_max_weight: float`, `shrinkage_minutes: float`, `max_paid_hits: int`, `hit_cost: int`, `bench_weight: list[float]`, `odds_provider: str | None`, `cache_ttl_hours: int`, `cache_ttl_matchday_hours: int`, `entry_id: int | None`, `free_transfers: int`; and `load_config(path: Path) -> Config`

- [ ] **Step 1: Initialise git and write project files**

```bash
cd "c:/Users/user/Documents/FPL Team Picker"
git init
```

`requirements.txt`:
```
pandas==2.2.1
numpy==1.26.4
scipy==1.14.1
pulp==2.9.0
pyarrow==18.1.0
requests==2.31.0
PyYAML>=6.0
pytest>=8.0
```

`.gitignore`:
```
data/cache/
data/processed/
data/archive/
data/state.json
__pycache__/
*.pyc
.pytest_cache/
```

`config.yaml`:
```yaml
budget: 100.0
horizon_gw: 5
risk:
  profile: balanced
  ownership_weight: 0.0
news:
  weight: 0.5
  max_age_hours: 48
model:
  form_half_life_gw: 3
  form_max_weight: 0.6
  shrinkage_minutes: 900
optimizer:
  max_paid_hits: 2
  hit_cost: 4
  bench_weight: [0.15, 0.10, 0.05, 0.02]
odds:
  provider: null
data:
  cache_ttl_hours: 6
  cache_ttl_matchday_hours: 1
entry_id: null
free_transfers: 1
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: pytest and PyYAML install successfully; the rest report "already satisfied".

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
import pytest
from fpl.config import load_config, Config


def test_loads_defaults_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "budget: 100.0\nhorizon_gw: 5\n"
        "risk: {profile: balanced, ownership_weight: 0.0}\n"
        "news: {weight: 0.5, max_age_hours: 48}\n"
        "model: {form_half_life_gw: 3, form_max_weight: 0.6, shrinkage_minutes: 900}\n"
        "optimizer: {max_paid_hits: 2, hit_cost: 4, bench_weight: [0.15, 0.1, 0.05, 0.02]}\n"
        "odds: {provider: null}\n"
        "data: {cache_ttl_hours: 6, cache_ttl_matchday_hours: 1}\n"
        "entry_id: null\nfree_transfers: 1\n"
    )
    c = load_config(p)
    assert isinstance(c, Config)
    assert c.budget == 100.0
    assert c.risk_profile == "balanced"
    assert c.bench_weight == [0.15, 0.1, 0.05, 0.02]
    assert c.entry_id is None
    assert c.free_transfers == 1


def test_missing_keys_get_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("budget: 95.0\n")
    c = load_config(p)
    assert c.budget == 95.0
    assert c.horizon_gw == 5
    assert c.hit_cost == 4


def test_rejects_free_transfers_above_cap(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("free_transfers: 6\n")
    with pytest.raises(ValueError, match="free_transfers"):
        load_config(p)


def test_rejects_unknown_risk_profile(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("risk: {profile: reckless}\n")
    with pytest.raises(ValueError, match="risk.profile"):
        load_config(p)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.config'`

- [ ] **Step 5: Write minimal implementation**

`fpl/__init__.py`: empty file.
`tests/__init__.py`: empty file.

`fpl/config.py`:
```python
"""Load and validate user preferences from config.yaml."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

VALID_PROFILES = {"balanced", "template", "differential"}
FT_CAP = 5


@dataclass
class Config:
    budget: float = 100.0
    horizon_gw: int = 5
    risk_profile: str = "balanced"
    ownership_weight: float = 0.0
    news_weight: float = 0.5
    news_max_age_hours: int = 48
    form_half_life_gw: float = 3.0
    form_max_weight: float = 0.6
    shrinkage_minutes: float = 900.0
    max_paid_hits: int = 2
    hit_cost: int = 4
    bench_weight: list[float] = field(default_factory=lambda: [0.15, 0.10, 0.05, 0.02])
    odds_provider: str | None = None
    cache_ttl_hours: int = 6
    cache_ttl_matchday_hours: int = 1
    entry_id: int | None = None
    free_transfers: int = 1


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    risk = raw.get("risk") or {}
    news = raw.get("news") or {}
    model = raw.get("model") or {}
    opt = raw.get("optimizer") or {}
    odds = raw.get("odds") or {}
    data = raw.get("data") or {}

    d = Config()
    cfg = Config(
        budget=float(raw.get("budget", d.budget)),
        horizon_gw=int(raw.get("horizon_gw", d.horizon_gw)),
        risk_profile=risk.get("profile", d.risk_profile),
        ownership_weight=float(risk.get("ownership_weight", d.ownership_weight)),
        news_weight=float(news.get("weight", d.news_weight)),
        news_max_age_hours=int(news.get("max_age_hours", d.news_max_age_hours)),
        form_half_life_gw=float(model.get("form_half_life_gw", d.form_half_life_gw)),
        form_max_weight=float(model.get("form_max_weight", d.form_max_weight)),
        shrinkage_minutes=float(model.get("shrinkage_minutes", d.shrinkage_minutes)),
        max_paid_hits=int(opt.get("max_paid_hits", d.max_paid_hits)),
        hit_cost=int(opt.get("hit_cost", d.hit_cost)),
        bench_weight=list(opt.get("bench_weight", d.bench_weight)),
        odds_provider=odds.get("provider", d.odds_provider),
        cache_ttl_hours=int(data.get("cache_ttl_hours", d.cache_ttl_hours)),
        cache_ttl_matchday_hours=int(data.get("cache_ttl_matchday_hours", d.cache_ttl_matchday_hours)),
        entry_id=raw.get("entry_id", d.entry_id),
        free_transfers=int(raw.get("free_transfers", d.free_transfers)),
    )

    if cfg.risk_profile not in VALID_PROFILES:
        raise ValueError(f"risk.profile must be one of {sorted(VALID_PROFILES)}, got {cfg.risk_profile!r}")
    if not 0 <= cfg.free_transfers <= FT_CAP:
        raise ValueError(f"free_transfers must be 0..{FT_CAP}, got {cfg.free_transfers}")
    if cfg.budget <= 0:
        raise ValueError(f"budget must be positive, got {cfg.budget}")
    if len(cfg.bench_weight) != 4:
        raise ValueError(f"optimizer.bench_weight needs exactly 4 values, got {len(cfg.bench_weight)}")
    return cfg
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt config.yaml fpl/ tests/ docs/
git commit -m "feat: project scaffold, config loader, design spec"
```

---

### Task 1: Snapshot cache

**Files:**
- Create: `fpl/data/__init__.py`, `fpl/data/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Cache(root: Path)` with methods `put(slug: str, payload) -> Path`, `newest(slug: str) -> tuple[dict | list, datetime] | None`, `get_fresh(slug: str, ttl_hours: float, now: datetime | None = None) -> dict | list | None`, `prune(slug: str, keep: int = 3) -> int`, and module function `is_matchday(fixtures: list[dict], now: datetime) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/test_cache.py`:
```python
from datetime import datetime, timedelta, timezone
import json
from fpl.data.cache import Cache, is_matchday

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_put_then_newest_roundtrips(tmp_path):
    c = Cache(tmp_path)
    c.put("bootstrap-static", {"a": 1})
    payload, ts = c.newest("bootstrap-static")
    assert payload == {"a": 1}
    assert ts.tzinfo is not None


def test_get_fresh_returns_payload_within_ttl(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1, 2], now=NOW - timedelta(hours=2))
    assert c.get_fresh("fixtures", ttl_hours=6, now=NOW) == [1, 2]


def test_get_fresh_returns_none_when_stale(tmp_path):
    c = Cache(tmp_path)
    c.put("fixtures", [1, 2], now=NOW - timedelta(hours=8))
    assert c.get_fresh("fixtures", ttl_hours=6, now=NOW) is None


def test_get_fresh_returns_none_when_absent(tmp_path):
    assert Cache(tmp_path).get_fresh("nope", ttl_hours=6, now=NOW) is None


def test_prune_keeps_newest_three(tmp_path):
    c = Cache(tmp_path)
    for h in range(6):
        c.put("bootstrap-static", {"n": h}, now=NOW - timedelta(hours=h))
    removed = c.prune("bootstrap-static", keep=3)
    assert removed == 3
    assert len(list(tmp_path.glob("bootstrap-static_*.json"))) == 3
    # newest survives
    assert c.newest("bootstrap-static")[0] == {"n": 0}


def test_prune_is_slug_scoped(tmp_path):
    c = Cache(tmp_path)
    for h in range(4):
        c.put("fixtures", {"n": h}, now=NOW - timedelta(hours=h))
    c.put("bootstrap-static", {"keep": True}, now=NOW)
    c.prune("fixtures", keep=3)
    assert c.newest("bootstrap-static")[0] == {"keep": True}


def test_is_matchday_true_when_fixture_kicks_off_today():
    fx = [{"kickoff_time": "2026-08-21T19:00:00Z"}]
    assert is_matchday(fx, NOW) is True


def test_is_matchday_false_on_empty_day():
    fx = [{"kickoff_time": "2026-08-22T19:00:00Z"}]
    assert is_matchday(fx, NOW) is False


def test_is_matchday_ignores_null_kickoffs():
    assert is_matchday([{"kickoff_time": None}], NOW) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.data'`

- [ ] **Step 3: Write minimal implementation**

`fpl/data/__init__.py`: empty file.

`fpl/data/cache.py`:
```python
"""Raw JSON snapshot cache with freshness checks and retention."""
from datetime import datetime, timezone
from pathlib import Path
import json

TS_FMT = "%Y%m%dT%H%M%SZ"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_matchday(fixtures: list[dict], now: datetime) -> bool:
    """True if any fixture kicks off on the same UTC date as `now`."""
    today = now.astimezone(timezone.utc).date()
    for f in fixtures or []:
        ko = f.get("kickoff_time")
        if not ko:
            continue
        if datetime.fromisoformat(ko.replace("Z", "+00:00")).astimezone(timezone.utc).date() == today:
            return True
    return False


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, slug: str) -> list[Path]:
        return sorted(self.root.glob(f"{slug}_*.json"), reverse=True)

    def put(self, slug: str, payload, now: datetime | None = None) -> Path:
        ts = (now or _now()).astimezone(timezone.utc)
        p = self.root / f"{slug}_{ts.strftime(TS_FMT)}.json"
        p.write_text(json.dumps(payload))
        return p

    def newest(self, slug: str):
        paths = self._paths(slug)
        if not paths:
            return None
        p = paths[0]
        ts = datetime.strptime(p.stem.split("_", 1)[1], TS_FMT).replace(tzinfo=timezone.utc)
        return json.loads(p.read_text()), ts

    def get_fresh(self, slug: str, ttl_hours: float, now: datetime | None = None):
        got = self.newest(slug)
        if got is None:
            return None
        payload, ts = got
        age_h = ((now or _now()) - ts).total_seconds() / 3600
        return payload if age_h < ttl_hours else None

    def prune(self, slug: str, keep: int = 3) -> int:
        paths = self._paths(slug)
        removed = 0
        for p in paths[keep:]:
            p.unlink()
            removed += 1
        return removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/data/ tests/test_cache.py
git commit -m "feat: snapshot cache with freshness and retention"
```

---

### Task 2: HTTP client

**Files:**
- Create: `fpl/data/client.py`, `tests/test_client.py`

**Interfaces:**
- Consumes: `Cache` from Task 1
- Produces: `FplClient(cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0, session=None)` with `bootstrap() -> dict`, `fixtures() -> list[dict]`, `element_summary(player_id: int) -> dict`, `entry(entry_id: int) -> dict`, `entry_history(entry_id: int) -> dict`, `entry_picks(entry_id: int, gw: int) -> dict`; plus attribute `stale: bool` set True when a fallback to cache occurred; module constant `BASE = "https://fantasy.premierleague.com/api/"`

- [ ] **Step 1: Write the failing test**

`tests/test_client.py`:
```python
import pytest
from fpl.data.cache import Cache
from fpl.data.client import FplClient, BASE


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes, fail=False):
        self.routes, self.fail, self.calls = routes, fail, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.fail:
            raise RuntimeError("network down")
        return FakeResponse(self.routes[url])


def test_bootstrap_fetches_and_caches(tmp_path):
    s = FakeSession({BASE + "bootstrap-static/": {"elements": []}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    assert c.bootstrap() == {"elements": []}
    assert c.stale is False
    assert len(s.calls) == 1


def test_second_call_uses_cache_not_network(tmp_path):
    s = FakeSession({BASE + "bootstrap-static/": {"elements": []}})
    cache = Cache(tmp_path)
    FplClient(cache, rate_limit_s=0, session=s).bootstrap()
    FplClient(cache, rate_limit_s=0, session=s).bootstrap()
    assert len(s.calls) == 1


def test_falls_back_to_stale_cache_on_failure(tmp_path):
    cache = Cache(tmp_path)
    cache.put("bootstrap-static", {"old": True})
    c = FplClient(cache, ttl_hours=0, rate_limit_s=0, session=FakeSession({}, fail=True))
    assert c.bootstrap() == {"old": True}
    assert c.stale is True


def test_raises_when_failure_and_no_cache(tmp_path):
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=FakeSession({}, fail=True))
    with pytest.raises(RuntimeError):
        c.bootstrap()


def test_never_requests_authenticated_endpoints(tmp_path):
    s = FakeSession({BASE + "fixtures/": [], BASE + "bootstrap-static/": {}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    c.fixtures()
    c.bootstrap()
    assert all("my-team" not in u for u in s.calls)


def test_entry_picks_url_shape(tmp_path):
    url = BASE + "entry/123/event/1/picks/"
    s = FakeSession({url: {"picks": []}})
    c = FplClient(Cache(tmp_path), rate_limit_s=0, session=s)
    assert c.entry_picks(123, 1) == {"picks": []}
    assert s.calls == [url]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.data.client'`

- [ ] **Step 3: Write minimal implementation**

`fpl/data/client.py`:
```python
"""Read-only HTTP client for the public FPL API.

Never calls any endpoint requiring a login/session cookie (e.g. my-team/),
and never issues a non-GET request.
"""
import time
import requests
from .cache import Cache

BASE = "https://fantasy.premierleague.com/api/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; fpl-team-picker/1.0)"}
FORBIDDEN = ("my-team",)


class FplClient:
    def __init__(self, cache: Cache, ttl_hours: float = 6, rate_limit_s: float = 1.0, session=None):
        self.cache = cache
        self.ttl_hours = ttl_hours
        self.rate_limit_s = rate_limit_s
        self.session = session or requests.Session()
        self.stale = False
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.rate_limit_s:
            delta = time.monotonic() - self._last_call
            if delta < self.rate_limit_s:
                time.sleep(self.rate_limit_s - delta)
            self._last_call = time.monotonic()

    def _get(self, path: str, slug: str):
        cached = self.cache.get_fresh(slug, self.ttl_hours)
        if cached is not None:
            return cached
        url = BASE + path
        if any(f in url for f in FORBIDDEN):
            raise ValueError(f"refusing to call authenticated endpoint: {url}")
        try:
            self._throttle()
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            fallback = self.cache.newest(slug)
            if fallback is None:
                raise
            self.stale = True
            return fallback[0]
        self.cache.put(slug, payload)
        self.cache.prune(slug, keep=3)
        return payload

    def bootstrap(self) -> dict:
        return self._get("bootstrap-static/", "bootstrap-static")

    def fixtures(self) -> list[dict]:
        return self._get("fixtures/", "fixtures")

    def element_summary(self, player_id: int) -> dict:
        return self._get(f"element-summary/{player_id}/", f"element-summary-{player_id}")

    def entry(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/", f"entry-{entry_id}")

    def entry_history(self, entry_id: int) -> dict:
        return self._get(f"entry/{entry_id}/history/", f"entry-history-{entry_id}")

    def entry_picks(self, entry_id: int, gw: int) -> dict:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/", f"entry-picks-{entry_id}-{gw}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_client.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/data/client.py tests/test_client.py
git commit -m "feat: read-only rate-limited FPL HTTP client"
```

---

### Task 3: Normalization and parquet store

**Files:**
- Create: `fpl/data/normalize.py`, `fpl/data/store.py`, `tests/test_normalize.py`

**Interfaces:**
- Consumes: raw payloads from Task 2
- Produces:
  - `normalize_teams(bootstrap: dict) -> pd.DataFrame` cols `team_id, name, short_name, strength_overall_home, strength_overall_away`
  - `normalize_players(bootstrap: dict) -> pd.DataFrame` cols `player_id, web_name, team_id, team, position, price, status, available, news, chance_of_playing, minutes, starts, total_points, goals_scored, assists, clean_sheets, goals_conceded, saves, bonus, bps, yellow_cards, red_cards, own_goals, expected_goals, expected_assists, expected_goals_conceded, selected_by_percent`
  - `normalize_fixtures(fixtures: list[dict]) -> pd.DataFrame` cols `fixture_id, event, team_h, team_a, team_h_difficulty, team_a_difficulty, kickoff_time, finished`
  - `history_past_frame(summaries: dict[int, dict]) -> pd.DataFrame` cols `player_id, season_name, total_points, minutes, starts, goals_scored, assists, clean_sheets, goals_conceded, saves, bonus, bps, yellow_cards, red_cards, own_goals, expected_goals, expected_assists, expected_goals_conceded, defensive_contribution, clearances_blocks_interceptions, tackles, recoveries`
  - `save_table(df, name: str, root: Path) -> Path` and `load_table(name: str, root: Path) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

`tests/test_normalize.py`:
```python
import pandas as pd
from fpl.data.normalize import (
    normalize_teams, normalize_players, normalize_fixtures, history_past_frame,
)
from fpl.data.store import save_table, load_table

BOOTSTRAP = {
    "teams": [
        {"id": 1, "name": "Arsenal", "short_name": "ARS",
         "strength_overall_home": 4, "strength_overall_away": 5},
        {"id": 7, "name": "Liverpool", "short_name": "LIV",
         "strength_overall_home": 5, "strength_overall_away": 4},
    ],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"},
        {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"},
        {"id": 4, "singular_name_short": "FWD"},
    ],
    "elements": [
        {"id": 12, "web_name": "Saka", "team": 1, "element_type": 3, "now_cost": 95,
         "status": "a", "news": "", "chance_of_playing_next_round": None,
         "minutes": 2218, "starts": 25, "total_points": 157, "goals_scored": 7,
         "assists": 10, "clean_sheets": 12, "goals_conceded": 16, "saves": 0,
         "bonus": 18, "bps": 570, "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
         "expected_goals": "7.57", "expected_assists": "7.16",
         "expected_goals_conceded": "15.57", "selected_by_percent": "11.2"},
        {"id": 99, "web_name": "Crock", "team": 7, "element_type": 2, "now_cost": 45,
         "status": "i", "news": "Knee injury - expected back 05 Sep",
         "chance_of_playing_next_round": 0,
         "minutes": 900, "starts": 10, "total_points": 40, "goals_scored": 0,
         "assists": 1, "clean_sheets": 4, "goals_conceded": 12, "saves": 0,
         "bonus": 2, "bps": 180, "yellow_cards": 1, "red_cards": 0, "own_goals": 0,
         "expected_goals": "0.40", "expected_assists": "0.90",
         "expected_goals_conceded": "13.10", "selected_by_percent": "0.4"},
    ],
}


def test_price_converted_from_tenths():
    df = normalize_players(BOOTSTRAP)
    assert df.loc[df.player_id == 12, "price"].iloc[0] == 9.5
    assert df.loc[df.player_id == 99, "price"].iloc[0] == 4.5


def test_ids_resolved_to_names():
    df = normalize_players(BOOTSTRAP)
    row = df[df.player_id == 12].iloc[0]
    assert row["team"] == "Arsenal"
    assert row["position"] == "MID"


def test_availability_flagged_from_status():
    df = normalize_players(BOOTSTRAP).set_index("player_id")
    assert bool(df.loc[12, "available"]) is True
    assert bool(df.loc[99, "available"]) is False
    assert "Knee injury" in df.loc[99, "news"]


def test_expected_stats_are_numeric():
    df = normalize_players(BOOTSTRAP)
    assert df["expected_goals"].dtype.kind == "f"
    assert df.loc[df.player_id == 12, "expected_goals"].iloc[0] == 7.57


def test_normalize_teams_shape():
    df = normalize_teams(BOOTSTRAP)
    assert list(df.columns[:3]) == ["team_id", "name", "short_name"]
    assert len(df) == 2


def test_normalize_fixtures_keeps_difficulty():
    fx = [{"id": 1, "event": 1, "team_h": 1, "team_a": 7, "team_h_difficulty": 2,
           "team_a_difficulty": 5, "kickoff_time": "2026-08-21T19:00:00Z", "finished": False}]
    df = normalize_fixtures(fx)
    assert df.loc[0, "team_h_difficulty"] == 2
    assert df.loc[0, "event"] == 1


def test_history_past_frame_flattens_seasons():
    summaries = {12: {"history_past": [
        {"season_name": "2024/25", "total_points": 127, "minutes": 2000, "starts": 22,
         "goals_scored": 6, "assists": 9, "clean_sheets": 10, "goals_conceded": 20,
         "saves": 0, "bonus": 12, "bps": 400, "yellow_cards": 1, "red_cards": 0,
         "own_goals": 0, "expected_goals": "6.0", "expected_assists": "6.5",
         "expected_goals_conceded": "18.0", "defensive_contribution": 150,
         "clearances_blocks_interceptions": 25, "tackles": 35, "recoveries": 90},
        {"season_name": "2025/26", "total_points": 157, "minutes": 2218, "starts": 25,
         "goals_scored": 7, "assists": 10, "clean_sheets": 12, "goals_conceded": 16,
         "saves": 0, "bonus": 18, "bps": 570, "yellow_cards": 2, "red_cards": 0,
         "own_goals": 0, "expected_goals": "7.57", "expected_assists": "7.16",
         "expected_goals_conceded": "15.57", "defensive_contribution": 184,
         "clearances_blocks_interceptions": 28, "tackles": 40, "recoveries": 116},
    ]}}
    df = history_past_frame(summaries)
    assert len(df) == 2
    assert set(df.player_id) == {12}
    assert df[df.season_name == "2025/26"]["defensive_contribution"].iloc[0] == 184


def test_store_roundtrip(tmp_path):
    df = normalize_players(BOOTSTRAP)
    save_table(df, "players", tmp_path)
    out = load_table("players", tmp_path)
    assert len(out) == len(df)
    assert out.loc[out.player_id == 12, "price"].iloc[0] == 9.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.data.normalize'`

- [ ] **Step 3: Write minimal implementation**

`fpl/data/normalize.py`:
```python
"""Raw FPL JSON -> tidy DataFrames. Resolves IDs, converts price, flags availability."""
import pandas as pd

AVAILABLE = "a"
FLOAT_COLS = ["expected_goals", "expected_assists", "expected_goals_conceded"]
PLAYER_INT_COLS = [
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "yellow_cards", "red_cards", "own_goals",
]
PAST_COLS = PLAYER_INT_COLS + [
    "defensive_contribution", "clearances_blocks_interceptions", "tackles", "recoveries",
]


def normalize_teams(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"])
    df = df.rename(columns={"id": "team_id"})
    return df[["team_id", "name", "short_name", "strength_overall_home", "strength_overall_away"]]


def normalize_players(bootstrap: dict) -> pd.DataFrame:
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    df = pd.DataFrame(bootstrap["elements"])
    df = df.rename(columns={"id": "player_id", "team": "team_id",
                            "chance_of_playing_next_round": "chance_of_playing"})
    df["team"] = df["team_id"].map(teams)
    df["position"] = df["element_type"].map(positions)
    df["price"] = df["now_cost"] / 10.0
    df["available"] = df["status"] == AVAILABLE
    df["news"] = df["news"].fillna("")
    for c in FLOAT_COLS + ["selected_by_percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in PLAYER_INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    cols = ["player_id", "web_name", "team_id", "team", "position", "price", "status",
            "available", "news", "chance_of_playing"] + PLAYER_INT_COLS + FLOAT_COLS + \
           ["selected_by_percent"]
    return df[cols].reset_index(drop=True)


def normalize_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(fixtures).rename(columns={"id": "fixture_id"})
    cols = ["fixture_id", "event", "team_h", "team_a", "team_h_difficulty",
            "team_a_difficulty", "kickoff_time", "finished"]
    return df[cols].reset_index(drop=True)


def history_past_frame(summaries: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for pid, summary in summaries.items():
        for season in summary.get("history_past", []):
            row = {"player_id": pid, "season_name": season["season_name"]}
            for c in PAST_COLS:
                row[c] = pd.to_numeric(season.get(c, 0), errors="coerce")
            for c in FLOAT_COLS:
                row[c] = pd.to_numeric(season.get(c, 0.0), errors="coerce")
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["player_id", "season_name"] + PAST_COLS + FLOAT_COLS)
    return pd.DataFrame(rows).fillna(0).reset_index(drop=True)
```

`fpl/data/store.py`:
```python
"""Parquet persistence for normalized tables."""
from pathlib import Path
import pandas as pd


def save_table(df: pd.DataFrame, name: str, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_table(name: str, root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / f"{name}.parquet")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/data/normalize.py fpl/data/store.py tests/test_normalize.py
git commit -m "feat: normalization layer and parquet store"
```

---

### Task 4: Team strength ratings

**Files:**
- Create: `fpl/model/__init__.py`, `fpl/model/strength.py`, `tests/test_strength.py`

**Interfaces:**
- Consumes: `normalize_players`, `normalize_teams`, `normalize_fixtures` output (Task 3)
- Produces: `team_ratings(players: pd.DataFrame, teams: pd.DataFrame, odds_provider=None) -> pd.DataFrame` cols `team_id, att, dfn, confidence` where `att`/`dfn` are multiplicative factors centred on 1.0 (`att` > 1 = scores more than average; `dfn` > 1 = concedes more than average), and `confidence` ∈ {`high`, `low`}. Also `HOME_ATT = 1.10`, `AWAY_ATT = 0.90` module constants.

**Why derived:** FPL's `strength_attack_*` / `strength_defence_*` fields are all `0` in the live 2026/27 data (spec §2), so ratings must be computed from last season's goals for/against, which survive in the bootstrap carryover.

- [ ] **Step 1: Write the failing test**

`tests/test_strength.py`:
```python
import pandas as pd
from fpl.model.strength import team_ratings, HOME_ATT, AWAY_ATT

TEAMS = pd.DataFrame({
    "team_id": [1, 2, 3],
    "name": ["Strong", "Average", "Promoted"],
    "short_name": ["STR", "AVG", "PRO"],
    "strength_overall_home": [5, 3, 2],
    "strength_overall_away": [5, 3, 2],
})

# Strong: 90 goals for, 20 against. Average: 50/50. Promoted: no PL history (all zero).
PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3],
    "team_id": [1, 2, 3],
    "goals_scored": [90, 50, 0],
    "goals_conceded": [20, 50, 0],
    "minutes": [3000, 3000, 0],
})


def test_ratings_centred_on_one():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert 0.9 < r.loc[2, "att"] < 1.1
    assert 0.9 < r.loc[2, "dfn"] < 1.1


def test_strong_team_has_higher_attack_and_lower_defence_factor():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[1, "att"] > r.loc[2, "att"]
    assert r.loc[1, "dfn"] < r.loc[2, "dfn"]


def test_promoted_team_falls_back_and_is_low_confidence():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[3, "confidence"] == "low"
    assert r.loc[3, "att"] > 0
    assert r.loc[3, "dfn"] > 0
    # weakest overall strength -> worst attack of the three
    assert r.loc[3, "att"] < r.loc[2, "att"]


def test_established_teams_are_high_confidence():
    r = team_ratings(PLAYERS, TEAMS).set_index("team_id")
    assert r.loc[1, "confidence"] == "high"


def test_odds_provider_overrides_promoted_rating():
    class FakeOdds:
        def team_factors(self, team_ids):
            return {3: (1.25, 0.80)}

    r = team_ratings(PLAYERS, TEAMS, odds_provider=FakeOdds()).set_index("team_id")
    assert r.loc[3, "att"] == 1.25
    assert r.loc[3, "dfn"] == 0.80
    assert r.loc[3, "confidence"] == "high"


def test_home_away_constants_are_symmetric_about_one():
    assert HOME_ATT > 1.0 > AWAY_ATT
    assert round((HOME_ATT + AWAY_ATT) / 2, 6) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strength.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/__init__.py`: empty file.

`fpl/model/strength.py`:
```python
"""Derived team attack/defence ratings.

FPL's strength_attack_* and strength_defence_* fields are zeroed in the live
2026/27 data, so ratings are computed from last season's goals for/against
(carried over in bootstrap). Teams with no PL history fall back to a prior
scaled from strength_overall_*, or to an optional odds provider.
"""
import pandas as pd

HOME_ATT = 1.10
AWAY_ATT = 0.90
MIN_MINUTES = 1  # a team with zero recorded minutes has no PL history


def _prior_from_overall(strength: float) -> float:
    """Map FPL's 1-5 overall strength onto a multiplicative factor near 1.0."""
    return 0.70 + 0.15 * float(strength)


def team_ratings(players: pd.DataFrame, teams: pd.DataFrame, odds_provider=None) -> pd.DataFrame:
    agg = players.groupby("team_id").agg(
        gf=("goals_scored", "sum"),
        ga=("goals_conceded", "sum"),
        mins=("minutes", "sum"),
    )
    df = teams[["team_id", "strength_overall_home", "strength_overall_away"]].merge(
        agg, left_on="team_id", right_index=True, how="left"
    ).fillna({"gf": 0, "ga": 0, "mins": 0})

    established = df["mins"] >= MIN_MINUTES
    mean_gf = df.loc[established, "gf"].mean() if established.any() else 1.0
    mean_ga = df.loc[established, "ga"].mean() if established.any() else 1.0

    att, dfn, conf = [], [], []
    for _, row in df.iterrows():
        if row["mins"] >= MIN_MINUTES and mean_gf > 0 and mean_ga > 0:
            att.append(row["gf"] / mean_gf)
            dfn.append(row["ga"] / mean_ga)
            conf.append("high")
        else:
            overall = (row["strength_overall_home"] + row["strength_overall_away"]) / 2
            a = _prior_from_overall(overall)
            att.append(a)
            dfn.append(2.0 - a)  # weak attack prior implies weak defence
            conf.append("low")
    df["att"], df["dfn"], df["confidence"] = att, dfn, conf

    if odds_provider is not None:
        factors = odds_provider.team_factors(list(df["team_id"]))
        for team_id, (a, d) in factors.items():
            mask = df["team_id"] == team_id
            df.loc[mask, ["att", "dfn", "confidence"]] = [a, d, "high"]

    return df[["team_id", "att", "dfn", "confidence"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_strength.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/model/ tests/test_strength.py
git commit -m "feat: derived team attack/defence ratings with odds hook"
```

---

### Task 5: Fixture layer — per-fixture difficulty, clean sheets, DGW/BGW

**Files:**
- Create: `fpl/model/fixtures.py`, `tests/test_fixtures.py`

**Interfaces:**
- Consumes: `normalize_fixtures` (Task 3), `team_ratings` (Task 4)
- Produces: `team_fixture_frame(fixtures: pd.DataFrame, ratings: pd.DataFrame, from_event: int, horizon: int) -> pd.DataFrame` with one row per (team, fixture) and cols `team_id, event, fixture_id, opponent_id, is_home, xgc, p_cs, att_mult`; and `fixture_counts(fixtures: pd.DataFrame, team_ids, from_event: int, horizon: int) -> pd.DataFrame` cols `team_id, event, n_fixtures` (0 = blank, 2+ = double)

**Maths:** `xgc = dfn_own × att_opp × (HOME_ATT if home else AWAY_ATT)`; `p_cs = exp(-xgc)`; `att_mult = att_own × dfn_opp × (HOME_ATT if home else AWAY_ATT)`.

- [ ] **Step 1: Write the failing test**

`tests/test_fixtures.py`:
```python
import math
import pandas as pd
from fpl.model.fixtures import team_fixture_frame, fixture_counts

RATINGS = pd.DataFrame({
    "team_id": [1, 2, 3],
    "att": [1.5, 1.0, 0.5],
    "dfn": [0.5, 1.0, 1.5],
    "confidence": ["high", "high", "high"],
})

FIX = pd.DataFrame({
    "fixture_id": [101, 102, 103, 104],
    "event": [1, 2, 2, 3],
    "team_h": [1, 1, 2, 2],
    "team_a": [2, 3, 3, 1],
    "team_h_difficulty": [3, 2, 2, 4],
    "team_a_difficulty": [4, 5, 4, 2],
    "kickoff_time": ["2026-08-21T19:00:00Z"] * 4,
    "finished": [False] * 4,
})


def test_each_fixture_yields_two_team_rows():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    assert len(df) == 8
    assert set(df[df.fixture_id == 101].team_id) == {1, 2}


def test_home_flag_and_opponent_resolved():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    row = df[(df.fixture_id == 101) & (df.team_id == 1)].iloc[0]
    assert bool(row.is_home) is True
    assert row.opponent_id == 2


def test_clean_sheet_probability_is_exp_neg_xgc():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3)
    row = df[(df.fixture_id == 101) & (df.team_id == 1)].iloc[0]
    assert math.isclose(row.p_cs, math.exp(-row.xgc), rel_tol=1e-9)
    assert 0 < row.p_cs < 1


def test_strong_defence_vs_weak_attack_has_higher_cs_probability():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=3).set_index(
        ["fixture_id", "team_id"])
    # team 1 (dfn 0.5) at home vs team 2 attack; team 2 (dfn 1.0) away vs team 1 attack
    assert df.loc[(101, 1), "p_cs"] > df.loc[(101, 2), "p_cs"]


def test_horizon_filters_events():
    df = team_fixture_frame(FIX, RATINGS, from_event=1, horizon=1)
    assert set(df.event) == {1}


def test_fixture_counts_detects_double_gameweek():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3).set_index(
        ["team_id", "event"])
    assert counts.loc[(1, 2), "n_fixtures"] == 1
    assert counts.loc[(3, 2), "n_fixtures"] == 2  # team 3 plays twice in event 2


def test_fixture_counts_detects_blank_gameweek():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3).set_index(
        ["team_id", "event"])
    assert counts.loc[(3, 1), "n_fixtures"] == 0  # team 3 has no event-1 fixture
    assert counts.loc[(3, 3), "n_fixtures"] == 0


def test_fixture_counts_covers_every_team_event_pair():
    counts = fixture_counts(FIX, [1, 2, 3], from_event=1, horizon=3)
    assert len(counts) == 9  # 3 teams x 3 events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model.fixtures'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/fixtures.py`:
```python
"""Per-fixture difficulty, clean-sheet probability, and DGW/BGW detection."""
import numpy as np
import pandas as pd
from .strength import HOME_ATT, AWAY_ATT


def team_fixture_frame(fixtures: pd.DataFrame, ratings: pd.DataFrame,
                       from_event: int, horizon: int) -> pd.DataFrame:
    events = range(from_event, from_event + horizon)
    fx = fixtures[fixtures["event"].isin(events)]
    r = ratings.set_index("team_id")

    rows = []
    for _, f in fx.iterrows():
        for team_id, opp_id, is_home in (
            (f["team_h"], f["team_a"], True),
            (f["team_a"], f["team_h"], False),
        ):
            if team_id not in r.index or opp_id not in r.index:
                continue
            venue = HOME_ATT if is_home else AWAY_ATT
            xgc = float(r.loc[team_id, "dfn"]) * float(r.loc[opp_id, "att"]) / venue
            rows.append({
                "team_id": int(team_id),
                "event": int(f["event"]),
                "fixture_id": int(f["fixture_id"]),
                "opponent_id": int(opp_id),
                "is_home": is_home,
                "xgc": xgc,
                "p_cs": float(np.exp(-xgc)),
                "att_mult": float(r.loc[team_id, "att"]) * float(r.loc[opp_id, "dfn"]) * venue,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


def fixture_counts(fixtures: pd.DataFrame, team_ids, from_event: int,
                   horizon: int) -> pd.DataFrame:
    events = list(range(from_event, from_event + horizon))
    fx = fixtures[fixtures["event"].isin(events)]
    played = {}
    for _, f in fx.iterrows():
        for t in (f["team_h"], f["team_a"]):
            played[(int(t), int(f["event"]))] = played.get((int(t), int(f["event"])), 0) + 1
    rows = [
        {"team_id": int(t), "event": int(e), "n_fixtures": played.get((int(t), int(e)), 0)}
        for t in team_ids for e in events
    ]
    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/model/fixtures.py tests/test_fixtures.py
git commit -m "feat: fixture difficulty, clean-sheet probability, DGW/BGW detection"
```

---

### Task 6: Minutes model

**Files:**
- Create: `fpl/model/minutes.py`, `tests/test_minutes.py`

**Interfaces:**
- Consumes: `normalize_players` (Task 3), `Config` (Task 0)
- Produces: `minutes_model(players: pd.DataFrame, cfg: Config, news: dict[int, dict] | None = None) -> pd.DataFrame` cols `player_id, p_start, p_play, p_60, e_minutes, confidence, flags` where `flags` is a `list[str]`. Module constants `M_START = 80.0`, `M_SUB = 20.0`, `P_SUB_APPEAR = 0.35`, `TEAM_GAMES = 38`.

**Rules (spec §6.3):** `p_start` from last season `starts / TEAM_GAMES`, shrunk by minutes. Hard override: `status` in `i`/`s`/`u` → `p_start = 0`; `d` → scale by `chance_of_playing / 100`. Players with zero PL minutes get a price-percentile prior within position, `confidence: low`, and an explicit flag. `news` dict maps `player_id -> {"p_start_override": float, "note": str, "source": str}` and is blended by `cfg.news_weight`.

- [ ] **Step 1: Write the failing test**

`tests/test_minutes.py`:
```python
import pandas as pd
from fpl.config import Config
from fpl.model.minutes import minutes_model, M_START, M_SUB

CFG = Config(shrinkage_minutes=900, news_weight=0.5)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "web_name": ["Nailed", "Rotation", "Injured", "Doubt", "NewSigning"],
    "position": ["MID", "MID", "DEF", "FWD", "MID"],
    "team_id": [1, 1, 2, 2, 3],
    "price": [9.0, 5.0, 4.5, 7.0, 8.5],
    "status": ["a", "a", "i", "d", "a"],
    "chance_of_playing": [None, None, 0, 25, None],
    "news": ["", "", "Knee injury", "Knock - 25% chance", ""],
    "available": [True, True, False, False, True],
    "minutes": [3200, 900, 2000, 2500, 0],
    "starts": [36, 8, 24, 29, 0],
})


def test_nailed_starter_has_high_p_start():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.75


def test_rotation_risk_has_lower_p_start_than_nailed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[2, "p_start"] < df.loc[1, "p_start"]


def test_injured_player_is_zeroed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[3, "p_start"] == 0.0
    assert df.loc[3, "e_minutes"] == 0.0
    assert any("unavailable" in f.lower() or "injur" in f.lower() for f in df.loc[3, "flags"])


def test_doubtful_player_scaled_by_chance_of_playing():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 < df.loc[4, "p_start"] < 0.4
    assert any("25%" in f for f in df.loc[4, "flags"])


def test_new_signing_gets_price_prior_and_low_confidence():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[5, "confidence"] == "low"
    assert df.loc[5, "p_start"] > 0
    assert any("limited data" in f.lower() for f in df.loc[5, "flags"])


def test_expected_minutes_bounded_by_start_and_sub_values():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 <= df.loc[1, "e_minutes"] <= M_START + M_SUB
    assert df.loc[1, "e_minutes"] > df.loc[2, "e_minutes"]


def test_p_60_never_exceeds_p_play():
    df = minutes_model(PLAYERS, CFG)
    assert (df["p_60"] <= df["p_play"] + 1e-9).all()


def test_news_override_blended_by_weight_and_flagged():
    news = {2: {"p_start_override": 1.0, "note": "confirmed to start", "source": "example.com"}}
    base = minutes_model(PLAYERS, CFG).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, CFG, news=news).set_index("player_id")
    assert out.loc[2, "p_start"] > base
    assert any("example.com" in f for f in out.loc[2, "flags"])


def test_news_weight_zero_ignores_news():
    news = {2: {"p_start_override": 1.0, "note": "confirmed", "source": "example.com"}}
    cfg = Config(shrinkage_minutes=900, news_weight=0.0)
    base = minutes_model(PLAYERS, cfg).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, cfg, news=news).set_index("player_id").loc[2, "p_start"]
    assert abs(out - base) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_minutes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model.minutes'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/minutes.py`:
```python
"""Minutes model: probability of starting, appearing, and reaching 60 minutes."""
import pandas as pd

M_START = 80.0     # typical minutes when a player starts
M_SUB = 20.0       # typical minutes when a player comes off the bench
P_SUB_APPEAR = 0.35  # chance a non-starter appears at all
TEAM_GAMES = 38
UNAVAILABLE = {"i", "s", "u"}
DOUBTFUL = "d"


def _price_prior(price: float, position: str) -> float:
    """Prior p_start for a player with no PL history, from FPL's own pricing signal."""
    floors = {"GKP": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
    floor = floors.get(position, 4.5)
    return float(min(0.85, max(0.15, (price - floor) / 6.0 + 0.25)))


def minutes_model(players: pd.DataFrame, cfg, news: dict[int, dict] | None = None) -> pd.DataFrame:
    news = news or {}
    k = float(cfg.shrinkage_minutes)
    pos_mean = (players["starts"] / TEAM_GAMES).mean()

    rows = []
    for _, p in players.iterrows():
        flags: list[str] = []
        confidence = "high"
        minutes = float(p["minutes"])

        if minutes <= 0:
            p_start = _price_prior(float(p["price"]), p["position"])
            confidence = "low"
            flags.append(
                f"Limited data: no Premier League minutes on record — "
                f"start probability inferred from price (£{p['price']}m)"
            )
        else:
            raw = float(p["starts"]) / TEAM_GAMES
            p_start = (minutes * raw + k * pos_mean) / (minutes + k)
            if minutes < 900:
                confidence = "medium"
                flags.append(f"Small sample: {int(minutes)} minutes last season")

        status = str(p["status"])
        if status in UNAVAILABLE:
            p_start = 0.0
            note = str(p["news"]).strip() or "unavailable"
            flags.append(f"Unavailable ({status}): {note}")
        elif status == DOUBTFUL:
            chance = p["chance_of_playing"]
            pct = 50.0 if pd.isna(chance) else float(chance)
            p_start *= pct / 100.0
            confidence = "low"
            note = str(p["news"]).strip()
            flags.append(f"Doubtful: {int(pct)}% chance of playing" + (f" — {note}" if note else ""))

        override = news.get(int(p["player_id"]))
        if override and cfg.news_weight > 0 and p_start > 0:
            w = float(cfg.news_weight)
            p_start = (1 - w) * p_start + w * float(override["p_start_override"])
            flags.append(f"Team news: {override['note']} (source: {override['source']})")

        p_start = float(min(1.0, max(0.0, p_start)))
        p_play = p_start + (1 - p_start) * P_SUB_APPEAR
        e_minutes = p_start * M_START + (p_play - p_start) * M_SUB
        p_60 = p_start  # reaching 60 minutes effectively requires starting

        rows.append({
            "player_id": int(p["player_id"]),
            "p_start": p_start,
            "p_play": p_play,
            "p_60": p_60,
            "e_minutes": e_minutes,
            "confidence": confidence,
            "flags": flags,
        })
    return pd.DataFrame(rows).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_minutes.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/model/minutes.py tests/test_minutes.py
git commit -m "feat: minutes model with availability overrides and news blending"
```

---

### Task 7: Per-90 rates with shrinkage, and the form blend

**Files:**
- Create: `fpl/model/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: `normalize_players` (Task 3), `Config` (Task 0)
- Produces:
  - `per90_rates(players: pd.DataFrame, cfg: Config) -> pd.DataFrame` cols `player_id, xg90, xa90, bonus90, dc90, saves90, cards90` — all shrunk toward the position mean
  - `ew_mean(values: list[float], half_life: float) -> float` — exponentially-weighted mean, most recent value last
  - `form_weight(gws_played: int, cfg: Config) -> float` — returns `min(1, gws_played/6) * cfg.form_max_weight`; **0 when `gws_played == 0`**
  - `blend_form(baseline: float, form_value: float, gws_played: int, cfg: Config) -> float`

**Shrinkage (spec §6.1):** `rate = (minutes × rate_player + k × rate_position_mean) / (minutes + k)`, with `k = cfg.shrinkage_minutes`.

**Critical:** at GW1 `gws_played = 0`, so `form_weight` must be exactly `0.0` — the model is 100% baseline. This is the single most important behaviour in this task.

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:
```python
import math
import pandas as pd
from fpl.config import Config
from fpl.model.scoring import per90_rates, ew_mean, form_weight, blend_form

CFG = Config(shrinkage_minutes=900, form_half_life_gw=3, form_max_weight=0.6)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3],
    "position": ["MID", "MID", "GKP"],
    "minutes": [3000, 90, 3000],
    "expected_goals": [15.0, 1.0, 0.0],
    "expected_assists": [10.0, 0.5, 0.0],
    "bonus": [30, 1, 12],
    "saves": [0, 0, 100],
    "yellow_cards": [4, 0, 1],
    "red_cards": [0, 0, 0],
    "defensive_contribution": [380, 12, 0],
})


def test_high_minutes_player_keeps_own_rate():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    # 15 xG over 3000 mins = 0.45/90; shrinkage pulls it only slightly
    assert 0.35 < r.loc[1, "xg90"] < 0.46


def test_tiny_sample_is_shrunk_toward_position_mean():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    # player 2 raw rate is 1.0 xG/90 but only 90 minutes -> heavily shrunk down
    assert r.loc[2, "xg90"] < 0.5


def test_shrinkage_is_monotonic_in_minutes():
    low = PLAYERS.copy()
    low.loc[low.player_id == 2, "minutes"] = 90
    high = PLAYERS.copy()
    high.loc[high.player_id == 2, "minutes"] = 3000
    high.loc[high.player_id == 2, "expected_goals"] = 33.3  # same 1.0/90 raw rate
    r_low = per90_rates(low, CFG).set_index("player_id").loc[2, "xg90"]
    r_high = per90_rates(high, CFG).set_index("player_id").loc[2, "xg90"]
    assert r_high > r_low


def test_saves_only_meaningful_for_keeper():
    r = per90_rates(PLAYERS, CFG).set_index("player_id")
    assert r.loc[3, "saves90"] > 1.0
    assert r.loc[1, "saves90"] == 0.0


def test_all_rates_non_negative():
    r = per90_rates(PLAYERS, CFG)
    for col in ["xg90", "xa90", "bonus90", "dc90", "saves90", "cards90"]:
        assert (r[col] >= 0).all()


def test_ew_mean_weights_recent_more():
    assert ew_mean([0, 0, 10], half_life=3) > ew_mean([10, 0, 0], half_life=3)


def test_ew_mean_of_constant_series_is_that_constant():
    assert math.isclose(ew_mean([5, 5, 5, 5], half_life=3), 5.0, rel_tol=1e-9)


def test_ew_mean_empty_returns_zero():
    assert ew_mean([], half_life=3) == 0.0


def test_form_weight_is_zero_at_gameweek_one():
    assert form_weight(0, CFG) == 0.0


def test_form_weight_ramps_then_caps():
    assert form_weight(3, CFG) == 0.5 * CFG.form_max_weight
    assert form_weight(6, CFG) == CFG.form_max_weight
    assert form_weight(20, CFG) == CFG.form_max_weight


def test_blend_form_returns_pure_baseline_at_gw1():
    assert blend_form(baseline=5.0, form_value=99.0, gws_played=0, cfg=CFG) == 5.0


def test_blend_form_moves_toward_form_as_season_progresses():
    out = blend_form(baseline=5.0, form_value=10.0, gws_played=6, cfg=CFG)
    assert math.isclose(out, 5.0 * 0.4 + 10.0 * 0.6, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model.scoring'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/scoring.py`:
```python
"""Per-90 scoring rates with shrinkage, and the baseline/form blend.

At GW1 there is no current-season gameweek data (spec §2), so form_weight
returns 0.0 and the model runs entirely on last season's baseline.
"""
import numpy as np
import pandas as pd

FORM_WINDOW_GWS = 6
RATE_SPECS = {
    "xg90": "expected_goals",
    "xa90": "expected_assists",
    "bonus90": "bonus",
    "dc90": "defensive_contribution",
    "saves90": "saves",
}


def per90_rates(players: pd.DataFrame, cfg) -> pd.DataFrame:
    k = float(cfg.shrinkage_minutes)
    df = players.copy()
    df["_cards"] = df["yellow_cards"] + 3 * df["red_cards"]
    mins = df["minutes"].astype(float)

    out = {"player_id": df["player_id"].astype(int)}
    specs = dict(RATE_SPECS, cards90="_cards")
    for name, source in specs.items():
        raw = np.where(mins > 0, df[source].astype(float) / np.maximum(mins, 1) * 90.0, 0.0)
        tmp = df.assign(_raw=raw, _mins=mins)
        pos_mean = tmp[tmp["_mins"] > 0].groupby("position")["_raw"].mean()
        fallback = float(tmp.loc[tmp["_mins"] > 0, "_raw"].mean() or 0.0)
        means = tmp["position"].map(pos_mean).fillna(fallback).astype(float)
        out[name] = (mins * raw + k * means) / (mins + k)
    return pd.DataFrame(out).reset_index(drop=True)


def ew_mean(values: list[float], half_life: float) -> float:
    """Exponentially-weighted mean. `values` is oldest-first, newest last."""
    if not values:
        return 0.0
    n = len(values)
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest has age 0
    weights = 0.5 ** (ages / float(half_life))
    return float(np.dot(weights, np.asarray(values, dtype=float)) / weights.sum())


def form_weight(gws_played: int, cfg) -> float:
    if gws_played <= 0:
        return 0.0
    return min(1.0, gws_played / FORM_WINDOW_GWS) * float(cfg.form_max_weight)


def blend_form(baseline: float, form_value: float, gws_played: int, cfg) -> float:
    w = form_weight(gws_played, cfg)
    return (1 - w) * float(baseline) + w * float(form_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/model/scoring.py tests/test_scoring.py
git commit -m "feat: shrunk per-90 rates and form blend (zero form weight at GW1)"
```

---

### Task 8: Expected bonus from BPS

**Files:**
- Create: `fpl/model/bps.py`, `tests/test_bps.py`

**Interfaces:**
- Consumes: `per90_rates` (Task 7), minutes frame (Task 6)
- Produces: `expected_bonus(rates: pd.DataFrame, minutes: pd.DataFrame, att_mult: float = 1.0) -> pd.Series` indexed by `player_id`

**Boundary (Global Constraints):** this models **bonus points only**. It must not award Defensive Contribution threshold points — Task 9 owns those. The two are additive.

- [ ] **Step 1: Write the failing test**

`tests/test_bps.py`:
```python
import pandas as pd
from fpl.model.bps import expected_bonus

RATES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "bonus90": [1.2, 0.1, 0.0],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "e_minutes": [85.0, 85.0, 0.0],
})


def test_high_bps_player_earns_more_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[1] > b.loc[2]


def test_zero_minutes_earns_no_bonus():
    b = expected_bonus(RATES, MINUTES)
    assert b.loc[3] == 0.0


def test_bonus_scales_with_expected_minutes():
    half = MINUTES.copy()
    half.loc[half.player_id == 1, "e_minutes"] = 42.5
    assert expected_bonus(RATES, half).loc[1] < expected_bonus(RATES, MINUTES).loc[1]


def test_favourable_fixture_multiplier_increases_bonus():
    base = expected_bonus(RATES, MINUTES).loc[1]
    boosted = expected_bonus(RATES, MINUTES, att_mult=1.4).loc[1]
    assert boosted > base


def test_bonus_never_exceeds_three_per_match():
    hot = pd.DataFrame({"player_id": [1], "bonus90": [99.0]})
    mins = pd.DataFrame({"player_id": [1], "e_minutes": [90.0]})
    assert expected_bonus(hot, mins, att_mult=3.0).loc[1] <= 3.0


def test_returns_series_indexed_by_player_id():
    b = expected_bonus(RATES, MINUTES)
    assert b.index.name == "player_id"
    assert set(b.index) == {1, 2, 3}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model.bps'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/bps.py`:
```python
"""Expected bonus points.

Models BONUS ONLY. Defensive Contribution threshold points are direct
scoring and are handled in model/xp.py — the same raw actions feed both
mechanisms, but they must not be awarded twice.
"""
import pandas as pd

MAX_BONUS_PER_MATCH = 3.0
FIXTURE_SENSITIVITY = 0.5  # bonus is less fixture-dependent than goals


def expected_bonus(rates: pd.DataFrame, minutes: pd.DataFrame,
                   att_mult: float = 1.0) -> pd.Series:
    df = rates[["player_id", "bonus90"]].merge(
        minutes[["player_id", "e_minutes"]], on="player_id", how="left"
    ).fillna({"e_minutes": 0.0})

    scale = 1.0 + (att_mult - 1.0) * FIXTURE_SENSITIVITY
    per_match = df["bonus90"] * (df["e_minutes"] / 90.0) * scale
    per_match = per_match.clip(lower=0.0, upper=MAX_BONUS_PER_MATCH)

    out = pd.Series(per_match.values, index=df["player_id"].astype(int))
    out.index.name = "player_id"
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bps.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/model/bps.py tests/test_bps.py
git commit -m "feat: expected bonus points from BPS tendency"
```

---

### Task 9: xP assembly — the model/optimizer contract

**Files:**
- Create: `fpl/model/xp.py`, `tests/test_xp.py`

**Interfaces:**
- Consumes: `per90_rates` (Task 7), `minutes_model` (Task 6), `team_fixture_frame` + `fixture_counts` (Task 5), `expected_bonus` (Task 8)
- Produces: **the contract frame** — `build_xp(players, rates, minutes, tfx, counts, cfg, from_event: int) -> pd.DataFrame` with exactly these columns: `player_id, web_name, team, position, price, xp_next1, xp_next5, p_start, e_minutes, confidence, flags`. Also exposes `p_dc_threshold(dc90: float, e_minutes: float, position: str) -> float` and `xp_for_fixture(...) -> float`.

**This is a swap seam.** `optimize/` reads only these columns. Any model producing this frame is a drop-in replacement — do not add optimizer-specific columns here.

**Scoring constants:** copy from Global Constraints. DC thresholds: DEF ≥10, MID/FWD ≥12, worth 2 pts, modelled per match as `p(Poisson(dc90 × e_minutes/90) ≥ threshold)` via `scipy.stats.poisson.sf(threshold-1, lam)`.

- [ ] **Step 1: Write the failing test**

`tests/test_xp.py`:
```python
import pandas as pd
import pytest
from fpl.config import Config
from fpl.model.xp import build_xp, p_dc_threshold, CONTRACT_COLUMNS

CFG = Config(horizon_gw=5)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3],
    "web_name": ["Striker", "Keeper", "Benched"],
    "team": ["Alpha", "Alpha", "Beta"],
    "team_id": [1, 1, 2],
    "position": ["FWD", "GKP", "MID"],
    "price": [11.0, 5.0, 4.5],
})
RATES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "xg90": [0.7, 0.0, 0.05],
    "xa90": [0.3, 0.0, 0.05],
    "bonus90": [0.8, 0.3, 0.0],
    "dc90": [1.0, 0.0, 3.0],
    "saves90": [0.0, 3.0, 0.0],
    "cards90": [0.1, 0.05, 0.1],
})
MINUTES = pd.DataFrame({
    "player_id": [1, 2, 3],
    "p_start": [0.95, 0.9, 0.0],
    "p_play": [0.97, 0.92, 0.0],
    "p_60": [0.95, 0.9, 0.0],
    "e_minutes": [82.0, 76.0, 0.0],
    "confidence": ["high", "high", "low"],
    "flags": [[], [], ["Unavailable (i): knee"]],
})
# Alpha plays every event; Beta blanks in event 1 and doubles in event 2.
TFX = pd.DataFrame({
    "team_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "event":   [1, 2, 3, 4, 5, 2, 2, 3, 4, 5],
    "fixture_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "opponent_id": [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    "is_home": [True, False, True, False, True, True, False, True, False, True],
    "xgc": [0.8] * 10,
    "p_cs": [0.45] * 10,
    "att_mult": [1.1] * 10,
})
COUNTS = pd.DataFrame(
    [{"team_id": 1, "event": e, "n_fixtures": 1} for e in range(1, 6)] +
    [{"team_id": 2, "event": 1, "n_fixtures": 0}, {"team_id": 2, "event": 2, "n_fixtures": 2}] +
    [{"team_id": 2, "event": e, "n_fixtures": 1} for e in range(3, 6)]
)


def _build():
    return build_xp(PLAYERS, RATES, MINUTES, TFX, COUNTS, CFG, from_event=1)


def test_contract_columns_exact():
    assert list(_build().columns) == CONTRACT_COLUMNS


def test_every_player_present():
    assert set(_build().player_id) == {1, 2, 3}


def test_striker_outscores_bench_player():
    df = _build().set_index("player_id")
    assert df.loc[1, "xp_next1"] > df.loc[3, "xp_next1"]


def test_zero_minutes_player_scores_zero():
    df = _build().set_index("player_id")
    assert df.loc[3, "xp_next1"] == 0.0
    assert df.loc[3, "xp_next5"] == 0.0


def test_blank_gameweek_gives_zero_for_that_event():
    # player 3 is on Beta, which has no event-1 fixture
    df = _build().set_index("player_id")
    assert df.loc[3, "xp_next1"] == 0.0


def test_double_gameweek_sums_both_fixtures():
    # Give the Beta player real minutes so the double is visible
    mins = MINUTES.copy()
    mins.loc[mins.player_id == 3, ["p_start", "p_play", "p_60", "e_minutes"]] = [0.9, 0.95, 0.9, 80.0]
    single = build_xp(PLAYERS, RATES, mins, TFX, COUNTS, CFG, from_event=2)
    alpha = single.set_index("player_id").loc[2, "xp_next1"]  # Alpha keeper, 1 fixture
    beta = single.set_index("player_id").loc[3, "xp_next1"]   # Beta mid, 2 fixtures
    per_fixture = build_xp(PLAYERS, RATES, mins, TFX, COUNTS, CFG, from_event=3)
    beta_single = per_fixture.set_index("player_id").loc[3, "xp_next1"]
    assert beta == pytest.approx(2 * beta_single, rel=0.02)
    assert alpha > 0


def test_xp_next5_at_least_xp_next1():
    df = _build()
    assert (df["xp_next5"] >= df["xp_next1"] - 1e-9).all()


def test_flags_propagate_from_minutes_model():
    df = _build().set_index("player_id")
    assert any("Unavailable" in f for f in df.loc[3, "flags"])


def test_keeper_earns_save_points():
    df = _build().set_index("player_id")
    assert df.loc[2, "xp_next1"] > 2.0  # appearance + saves + clean-sheet share


def test_dc_threshold_uses_position_specific_bar():
    # identical rate: defenders need 10, midfielders need 12 -> defender more likely
    assert p_dc_threshold(12.0, 90.0, "DEF") > p_dc_threshold(12.0, 90.0, "MID")


def test_dc_threshold_zero_when_no_minutes():
    assert p_dc_threshold(12.0, 0.0, "DEF") == 0.0


def test_dc_threshold_is_a_probability():
    for rate in (0.0, 5.0, 20.0):
        p = p_dc_threshold(rate, 90.0, "MID")
        assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_xp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.model.xp'`

- [ ] **Step 3: Write minimal implementation**

`fpl/model/xp.py`:
```python
"""Assemble expected points per player.

THIS MODULE DEFINES THE MODEL/OPTIMIZER CONTRACT. optimize/ consumes only
CONTRACT_COLUMNS. Any replacement model that emits this frame is a drop-in.

xP is computed per fixture and summed over the fixtures in an event, so
double gameweeks (2+ fixtures) and blanks (0 fixtures) fall out for free.
"""
import pandas as pd
from scipy.stats import poisson

from .bps import expected_bonus

CONTRACT_COLUMNS = [
    "player_id", "web_name", "team", "position", "price",
    "xp_next1", "xp_next5", "p_start", "e_minutes", "confidence", "flags",
]

GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
DC_PTS = 2
DC_THRESHOLD = {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12}
SAVES_PER_POINT = 3.0
CONCEDED_PENALTY_POSITIONS = {"GKP", "DEF"}


def p_dc_threshold(dc90: float, e_minutes: float, position: str) -> float:
    """Probability of hitting the Defensive Contribution threshold in one match."""
    if e_minutes <= 0 or dc90 <= 0:
        return 0.0
    threshold = DC_THRESHOLD.get(position, 12)
    lam = float(dc90) * float(e_minutes) / 90.0
    return float(poisson.sf(threshold - 1, lam))


def xp_for_fixture(rate_row, mins_row, fx_row, position: str, bonus: float) -> float:
    e_min = float(mins_row["e_minutes"])
    if e_min <= 0:
        return 0.0
    share = e_min / 90.0
    p_play, p_60 = float(mins_row["p_play"]), float(mins_row["p_60"])

    pts = p_play + p_60  # 1 pt for appearing, 2 for 60+
    pts += float(rate_row["xg90"]) * share * float(fx_row["att_mult"]) * GOAL_PTS[position]
    pts += float(rate_row["xa90"]) * share * float(fx_row["att_mult"]) * ASSIST_PTS
    pts += float(fx_row["p_cs"]) * CS_PTS[position] * p_60
    pts += p_dc_threshold(float(rate_row["dc90"]), e_min, position) * DC_PTS
    pts += bonus
    if position in CONCEDED_PENALTY_POSITIONS:
        pts -= 0.5 * float(fx_row["xgc"]) * share
    if position == "GKP":
        pts += float(rate_row["saves90"]) * share / SAVES_PER_POINT
    pts -= float(rate_row["cards90"]) * share
    return max(0.0, pts)


def build_xp(players: pd.DataFrame, rates: pd.DataFrame, minutes: pd.DataFrame,
             tfx: pd.DataFrame, counts: pd.DataFrame, cfg, from_event: int) -> pd.DataFrame:
    r = rates.set_index("player_id")
    m = minutes.set_index("player_id")
    horizon_events = list(range(from_event, from_event + cfg.horizon_gw))

    rows = []
    for _, p in players.iterrows():
        pid, pos, team_id = int(p["player_id"]), p["position"], int(p["team_id"])
        rate_row, mins_row = r.loc[pid], m.loc[pid]
        fixtures = tfx[tfx["team_id"] == team_id]

        per_event: dict[int, float] = {}
        for _, fx in fixtures.iterrows():
            event = int(fx["event"])
            if event not in horizon_events:
                continue
            bonus = float(expected_bonus(
                rates[rates.player_id == pid],
                minutes[minutes.player_id == pid],
                att_mult=float(fx["att_mult"]),
            ).loc[pid])
            per_event[event] = per_event.get(event, 0.0) + xp_for_fixture(
                rate_row, mins_row, fx, pos, bonus
            )

        rows.append({
            "player_id": pid,
            "web_name": p["web_name"],
            "team": p["team"],
            "position": pos,
            "price": float(p["price"]),
            "xp_next1": round(per_event.get(from_event, 0.0), 4),
            "xp_next5": round(sum(per_event.values()), 4),
            "p_start": float(mins_row["p_start"]),
            "e_minutes": float(mins_row["e_minutes"]),
            "confidence": mins_row["confidence"],
            "flags": list(mins_row["flags"]),
        })
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_xp.py -v`
Expected: 12 passed

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add fpl/model/xp.py tests/test_xp.py
git commit -m "feat: xP assembly with DC threshold, DGW/BGW handling, contract frame"
```

---

### Task 10: Historical archive loader and Tier 1 backtest

**Files:**
- Create: `fpl/backtest/__init__.py`, `fpl/data/archive.py`, `fpl/backtest/aggregate.py`, `tests/test_backtest_aggregate.py`

**Interfaces:**
- Consumes: `Cache` (Task 1), `history_past_frame` (Task 3)
- Produces:
  - `load_season_gws(season: str, cache: Cache, session=None) -> pd.DataFrame` — fetches `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv`, cached under slug `archive-{season}`. Columns kept: `element, name, position, team, GW, total_points, minutes, xP, expected_goals, expected_assists, bps, was_home, opponent_team`
  - `verify_archive_integrity(gw_df: pd.DataFrame, past_df: pd.DataFrame, season: str, tolerance: float = 0.02) -> dict` → `{"checked": int, "mismatched": int, "ok": bool}`
  - `walk_forward_aggregate(past: pd.DataFrame, cfg) -> dict` → `{"mae": float, "rmse": float, "spearman": float, "n": int, "naive_mae": float, "beats_naive": bool}`

**Archive is backtest-only.** It must never be imported by `run_gameweek.py`'s weekly path.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_aggregate.py`:
```python
import pandas as pd
from fpl.data.cache import Cache
from fpl.data.archive import load_season_gws, verify_archive_integrity, ARCHIVE_URL
from fpl.backtest.aggregate import walk_forward_aggregate
from fpl.config import Config

CSV = (
    "name,position,team,element,GW,total_points,minutes,xP,"
    "expected_goals,expected_assists,bps,was_home,opponent_team\n"
    "Saka,MID,Arsenal,12,1,8,90,5.2,0.4,0.3,32,True,7\n"
    "Saka,MID,Arsenal,12,2,2,90,4.8,0.2,0.1,14,False,3\n"
    "Rice,MID,Arsenal,13,1,6,90,4.1,0.1,0.2,28,True,7\n"
)


class FakeResp:
    status_code = 200
    text = CSV

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return FakeResp()


def test_load_season_parses_csv(tmp_path):
    df = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    assert len(df) == 3
    assert set(df.columns) >= {"element", "GW", "total_points", "minutes", "xP"}
    assert df.loc[df.element == 12, "total_points"].sum() == 10


def test_load_season_uses_cache_on_second_call(tmp_path):
    cache, s = Cache(tmp_path), FakeSession()
    load_season_gws("2025-26", cache, session=s)
    load_season_gws("2025-26", cache, session=s)
    assert len(s.calls) == 1
    assert "2025-26" in s.calls[0]
    assert s.calls[0].startswith(ARCHIVE_URL.split("{")[0])


def test_integrity_check_passes_when_totals_match(tmp_path):
    gw = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    past = pd.DataFrame({"player_id": [12, 13], "season_name": ["2025/26"] * 2,
                         "total_points": [10, 6], "minutes": [180, 90]})
    result = verify_archive_integrity(gw, past, "2025/26")
    assert result["ok"] is True
    assert result["mismatched"] == 0


def test_integrity_check_flags_mismatch(tmp_path):
    gw = load_season_gws("2025-26", Cache(tmp_path), session=FakeSession())
    past = pd.DataFrame({"player_id": [12, 13], "season_name": ["2025/26"] * 2,
                         "total_points": [999, 6], "minutes": [180, 90]})
    result = verify_archive_integrity(gw, past, "2025/26")
    assert result["ok"] is False
    assert result["mismatched"] == 1


def test_walk_forward_reports_error_and_naive_comparison():
    past = pd.DataFrame({
        "player_id": [1, 1, 2, 2, 3, 3],
        "season_name": ["2024/25", "2025/26"] * 3,
        "total_points": [100, 110, 50, 45, 200, 190],
        "minutes": [3000, 3000, 2000, 2000, 3200, 3100],
    })
    out = walk_forward_aggregate(past, Config())
    assert out["n"] == 3
    assert out["mae"] >= 0
    assert "naive_mae" in out
    assert isinstance(out["beats_naive"], bool)


def test_walk_forward_handles_players_with_one_season():
    past = pd.DataFrame({
        "player_id": [1, 2],
        "season_name": ["2025/26", "2025/26"],
        "total_points": [100, 50],
        "minutes": [3000, 2000],
    })
    out = walk_forward_aggregate(past, Config())
    assert out["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.data.archive'`

- [ ] **Step 3: Write minimal implementation**

`fpl/backtest/__init__.py`: empty file.

`fpl/data/archive.py`:
```python
"""Historical per-GW data (backtest only).

NEVER import this from the weekly pipeline. It exists solely as a cold-start
crutch for pre-season validation; from GW2 the project's own cached
element-summary/history is the GW-level source.
"""
from io import StringIO
import pandas as pd
import requests

from .cache import Cache

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
KEEP = ["element", "name", "position", "team", "GW", "total_points", "minutes",
        "xP", "expected_goals", "expected_assists", "bps", "was_home", "opponent_team"]


def load_season_gws(season: str, cache: Cache, session=None) -> pd.DataFrame:
    slug = f"archive-{season}"
    cached = cache.newest(slug)
    if cached is not None:
        return pd.DataFrame(cached[0])
    resp = (session or requests).get(ARCHIVE_URL.format(season=season), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df = df[[c for c in KEEP if c in df.columns]]
    cache.put(slug, df.to_dict("records"))
    cache.prune(slug, keep=3)
    return df


def verify_archive_integrity(gw_df: pd.DataFrame, past_df: pd.DataFrame,
                             season: str, tolerance: float = 0.02) -> dict:
    """Do per-GW rows sum to the API's season totals? If not, the archive is suspect."""
    totals = gw_df.groupby("element")["total_points"].sum()
    past = past_df[past_df["season_name"] == season].set_index("player_id")["total_points"]
    common = totals.index.intersection(past.index)
    checked = mismatched = 0
    for pid in common:
        checked += 1
        expected, actual = float(past.loc[pid]), float(totals.loc[pid])
        if abs(expected - actual) > max(1.0, tolerance * abs(expected)):
            mismatched += 1
    return {"checked": checked, "mismatched": mismatched,
            "ok": checked > 0 and mismatched == 0}
```

`fpl/backtest/aggregate.py`:
```python
"""Tier 1 backtest: multi-season walk-forward over history_past aggregates.

Validates the per-90 baseline and shrinkage. Cannot validate fixture
adjustment, form decay, or captaincy — that is Tier 2's job.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _pts90(row) -> float:
    return float(row["total_points"]) / max(float(row["minutes"]), 1.0) * 90.0


def walk_forward_aggregate(past: pd.DataFrame, cfg) -> dict:
    """Predict each season's points-per-90 from the player's prior seasons."""
    df = past.sort_values(["player_id", "season_name"]).copy()
    df["pts90"] = df.apply(_pts90, axis=1)

    preds, actuals = [], []
    for _, group in df.groupby("player_id"):
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        history = rows[:-1]
        target = rows[-1]
        mins = sum(float(r["minutes"]) for r in history)
        weighted = sum(_pts90(r) * float(r["minutes"]) for r in history)
        k = float(cfg.shrinkage_minutes)
        pop_mean = float(df["pts90"].mean())
        pred = (weighted + k * pop_mean) / (mins + k)
        preds.append(pred)
        actuals.append(float(target["pts90"]))

    if not preds:
        return {"mae": 0.0, "rmse": 0.0, "spearman": 0.0, "n": 0,
                "naive_mae": 0.0, "beats_naive": False}

    preds_a, actual_a = np.array(preds), np.array(actuals)
    mae = float(np.mean(np.abs(preds_a - actual_a)))
    rmse = float(np.sqrt(np.mean((preds_a - actual_a) ** 2)))
    rho = float(spearmanr(preds_a, actual_a).statistic) if len(preds_a) > 2 else 0.0
    naive = float(np.mean(np.abs(np.full_like(actual_a, actual_a.mean()) - actual_a)))
    return {"mae": mae, "rmse": rmse, "spearman": 0.0 if np.isnan(rho) else rho,
            "n": len(preds), "naive_mae": naive, "beats_naive": mae < naive}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_aggregate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/data/archive.py fpl/backtest/ tests/test_backtest_aggregate.py
git commit -m "feat: archive loader with integrity check and Tier 1 aggregate backtest"
```

---

### Task 11: Tier 2 per-GW backtest and the trust gate

**Files:**
- Create: `fpl/backtest/gw_level.py`, `tests/test_backtest_gw.py`

**Interfaces:**
- Consumes: `load_season_gws` (Task 10)
- Produces: `evaluate_predictions(pred: pd.Series, actual: pd.Series, positions: pd.Series) -> dict` → `{"mae", "rmse", "spearman_overall", "spearman_by_position": dict, "top20_overlap": float, "n"}`; `captaincy_hit_rate(pred_by_gw: dict[int, pd.Series], actual_by_gw: dict[int, pd.Series]) -> float`; `trust_gate(model: dict, naive: dict, fpl_xp: dict) -> dict` → `{"trusted": bool, "failures": list[str], "summary": str}`

**Gate rule (spec §7):** the model is trusted only if its **per-position Spearman** beats *both* the naive last-season-per-90 baseline and FPL's own `xP` column. Any position where it fails is named in `failures`.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_gw.py`:
```python
import pandas as pd
from fpl.backtest.gw_level import evaluate_predictions, captaincy_hit_rate, trust_gate

POS = pd.Series(["MID", "MID", "DEF", "DEF", "FWD", "FWD"], index=range(6))
ACTUAL = pd.Series([10, 2, 8, 1, 12, 3], index=range(6))
GOOD = pd.Series([9, 3, 7, 2, 11, 4], index=range(6))
BAD = pd.Series([2, 10, 1, 8, 3, 12], index=range(6))


def test_good_predictions_have_low_error():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert out["mae"] < 1.5
    assert out["n"] == 6


def test_good_predictions_rank_positively():
    assert evaluate_predictions(GOOD, ACTUAL, POS)["spearman_overall"] > 0.8


def test_inverted_predictions_rank_negatively():
    assert evaluate_predictions(BAD, ACTUAL, POS)["spearman_overall"] < 0


def test_spearman_reported_per_position():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert set(out["spearman_by_position"]) == {"MID", "DEF", "FWD"}


def test_top20_overlap_is_a_fraction():
    out = evaluate_predictions(GOOD, ACTUAL, POS)
    assert 0.0 <= out["top20_overlap"] <= 1.0


def test_captaincy_hit_rate_perfect_when_top_pick_is_top_scorer():
    pred = {1: pd.Series([5, 9], index=[10, 11]), 2: pd.Series([7, 2], index=[10, 11])}
    actual = {1: pd.Series([4, 12], index=[10, 11]), 2: pd.Series([9, 1], index=[10, 11])}
    assert captaincy_hit_rate(pred, actual) == 1.0


def test_captaincy_hit_rate_zero_when_always_wrong():
    pred = {1: pd.Series([9, 5], index=[10, 11])}
    actual = {1: pd.Series([1, 12], index=[10, 11])}
    assert captaincy_hit_rate(pred, actual) == 0.0


def test_trust_gate_passes_when_model_beats_both_baselines():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.5, "FWD": 0.55}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.2, "FWD": 0.25}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}
    out = trust_gate(model, naive, fpl)
    assert out["trusted"] is True
    assert out["failures"] == []


def test_trust_gate_fails_and_names_the_position():
    model = {"spearman_by_position": {"MID": 0.6, "DEF": 0.1, "FWD": 0.55}}
    naive = {"spearman_by_position": {"MID": 0.3, "DEF": 0.4, "FWD": 0.25}}
    fpl = {"spearman_by_position": {"MID": 0.4, "DEF": 0.3, "FWD": 0.35}}
    out = trust_gate(model, naive, fpl)
    assert out["trusted"] is False
    assert any("DEF" in f for f in out["failures"])
    assert "DEF" in out["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_gw.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.backtest.gw_level'`

- [ ] **Step 3: Write minimal implementation**

`fpl/backtest/gw_level.py`:
```python
"""Tier 2 backtest: per-gameweek accuracy and the model trust gate.

Validates what aggregates cannot: fixture adjustment, form decay, captaincy.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TOP_N = 20


def _rho(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return 0.0
    r = spearmanr(a.values, b.values).statistic
    return 0.0 if np.isnan(r) else float(r)


def evaluate_predictions(pred: pd.Series, actual: pd.Series,
                         positions: pd.Series) -> dict:
    df = pd.DataFrame({"pred": pred, "actual": actual, "pos": positions}).dropna()
    err = df["pred"] - df["actual"]
    n_top = min(TOP_N, len(df))
    top_pred = set(df["pred"].nlargest(n_top).index)
    top_actual = set(df["actual"].nlargest(n_top).index)
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman_overall": _rho(df["pred"], df["actual"]),
        "spearman_by_position": {
            pos: _rho(g["pred"], g["actual"]) for pos, g in df.groupby("pos")
        },
        "top20_overlap": len(top_pred & top_actual) / n_top if n_top else 0.0,
        "n": int(len(df)),
    }


def captaincy_hit_rate(pred_by_gw: dict[int, pd.Series],
                       actual_by_gw: dict[int, pd.Series]) -> float:
    hits = total = 0
    for gw, pred in pred_by_gw.items():
        actual = actual_by_gw.get(gw)
        if actual is None or pred.empty:
            continue
        total += 1
        if pred.idxmax() == actual.idxmax():
            hits += 1
    return hits / total if total else 0.0


def trust_gate(model: dict, naive: dict, fpl_xp: dict) -> dict:
    """Model is trusted only if per-position rank correlation beats BOTH baselines."""
    failures = []
    for pos, rho in model["spearman_by_position"].items():
        if rho <= naive["spearman_by_position"].get(pos, -1):
            failures.append(f"{pos}: rank correlation {rho:.3f} does not beat naive baseline")
        if rho <= fpl_xp["spearman_by_position"].get(pos, -1):
            failures.append(f"{pos}: rank correlation {rho:.3f} does not beat FPL's own xP")
    trusted = not failures
    summary = ("Model beats both baselines in every position — recommendations can be "
               "trusted at face value." if trusted else
               "LOW CONFIDENCE — " + "; ".join(failures))
    return {"trusted": trusted, "failures": failures, "summary": summary}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_gw.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/backtest/gw_level.py tests/test_backtest_gw.py
git commit -m "feat: per-GW backtest metrics and model trust gate"
```

---

### Task 12: Mode 1 — full squad build (joint MILP)

**Files:**
- Create: `fpl/optimize/__init__.py`, `fpl/optimize/squad.py`, `tests/test_squad.py`

**Interfaces:**
- Consumes: the contract frame from `build_xp` (Task 9), `Config` (Task 0)
- Produces: `@dataclass Squad(player_ids: list[int], starting_ids: list[int], total_cost: float, xp: float)` and `optimize_squad(xp_df: pd.DataFrame, cfg: Config, xp_col: str = "xp_next5", must_include: list[int] | None = None, banned: list[int] | None = None) -> Squad`

**Constraints — these tests are the highest-value in the codebase.** An invalid squad cannot be applied at all, so it is worse than no recommendation. Budget ≤ `cfg.budget`; 15 players as 2 GK / 5 DEF / 5 MID / 3 FWD; ≤3 per club; XI = 11 with 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD.

**Joint objective (spec §8):** `maximize Σ xp·start + bench_w · Σ xp·(squad − start)` where `bench_w = mean(cfg.bench_weight)`. A two-stage "pick 15 then pick 11" would overspend on bench players who score nothing. The per-slot weights in config are applied by `lineup.py` for reporting; the MILP uses their mean because exact bench ordering would require ordering variables for negligible gain.

- [ ] **Step 1: Write the failing test**

`tests/test_squad.py`:
```python
import pandas as pd
import pytest
from fpl.config import Config
from fpl.optimize.squad import optimize_squad, Squad

CFG = Config(budget=100.0)


def make_pool(n_per_team=8, n_teams=10):
    """A pool rich enough that a valid 15 always exists."""
    rows, pid = [], 1
    for t in range(n_teams):
        for i in range(n_per_team):
            pos = ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i % 8]
            rows.append({
                "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                "position": pos, "price": 4.0 + (i % 5) * 1.5,
                "xp_next1": 1.0 + (pid % 7) * 0.4, "xp_next5": 5.0 + (pid % 7) * 1.3,
                "p_start": 0.9, "e_minutes": 80.0, "confidence": "high", "flags": [],
            })
            pid += 1
    return pd.DataFrame(rows)


POOL = make_pool()


def test_returns_exactly_fifteen_players():
    s = optimize_squad(POOL, CFG)
    assert isinstance(s, Squad)
    assert len(s.player_ids) == 15
    assert len(set(s.player_ids)) == 15


def test_position_split_is_two_five_five_three():
    s = optimize_squad(POOL, CFG)
    picked = POOL[POOL.player_id.isin(s.player_ids)]
    counts = picked.position.value_counts().to_dict()
    assert counts["GKP"] == 2 and counts["DEF"] == 5
    assert counts["MID"] == 5 and counts["FWD"] == 3


def test_budget_never_exceeded():
    s = optimize_squad(POOL, CFG)
    assert s.total_cost <= CFG.budget + 1e-6


def test_max_three_players_per_club():
    s = optimize_squad(POOL, CFG)
    picked = POOL[POOL.player_id.isin(s.player_ids)]
    assert picked.team.value_counts().max() <= 3


def test_starting_eleven_is_valid_formation():
    s = optimize_squad(POOL, CFG)
    xi = POOL[POOL.player_id.isin(s.starting_ids)]
    assert len(s.starting_ids) == 11
    c = xi.position.value_counts().to_dict()
    assert c.get("GKP", 0) == 1
    assert 3 <= c.get("DEF", 0) <= 5
    assert 2 <= c.get("MID", 0) <= 5
    assert 1 <= c.get("FWD", 0) <= 3


def test_starters_are_a_subset_of_the_squad():
    s = optimize_squad(POOL, CFG)
    assert set(s.starting_ids).issubset(set(s.player_ids))


def test_tighter_budget_still_produces_valid_squad():
    s = optimize_squad(POOL, Config(budget=80.0))
    assert len(s.player_ids) == 15
    assert s.total_cost <= 80.0 + 1e-6


def test_known_optimum_on_small_pool():
    """Two clear tiers: the optimiser must take every premium it can afford."""
    rows = []
    pid = 1
    for t in range(6):
        for pos, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(count):
                premium = pid % 4 == 0
                rows.append({
                    "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                    "position": pos, "price": 4.0, "xp_next1": 1.0,
                    "xp_next5": 50.0 if premium else 1.0, "p_start": 0.9,
                    "e_minutes": 80.0, "confidence": "high", "flags": [],
                })
                pid += 1
    pool = pd.DataFrame(rows)
    s = optimize_squad(pool, Config(budget=100.0))
    picked = pool[pool.player_id.isin(s.player_ids)]
    # every pick is affordable at 4.0 x 15 = 60 <= 100, so it maximises premiums
    # subject to 3-per-club and the 2/5/5/3 split
    assert (picked.xp_next5 == 50.0).sum() >= 8


def test_banned_players_are_excluded():
    banned = list(POOL.player_id[:5])
    s = optimize_squad(POOL, CFG, banned=banned)
    assert not set(s.player_ids) & set(banned)


def test_must_include_players_are_selected():
    forced = [int(POOL.player_id.iloc[0]), int(POOL.player_id.iloc[4])]
    s = optimize_squad(POOL, CFG, must_include=forced)
    assert set(forced).issubset(set(s.player_ids))


def test_infeasible_budget_raises():
    pricey = POOL.copy()
    pricey["price"] = 20.0
    with pytest.raises(ValueError, match="infeasible"):
        optimize_squad(pricey, Config(budget=50.0))


def test_optimises_on_requested_horizon_column():
    short = optimize_squad(POOL, CFG, xp_col="xp_next1")
    long = optimize_squad(POOL, CFG, xp_col="xp_next5")
    assert len(short.player_ids) == len(long.player_ids) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_squad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.optimize'`

- [ ] **Step 3: Write minimal implementation**

`fpl/optimize/__init__.py`: empty file.

`fpl/optimize/squad.py`:
```python
"""Mode 1: build a full 15-man squad from scratch (initial team / wildcard / free hit).

Single joint MILP over squad and starting-XI membership. A two-stage
"pick 15 then pick 11" would spend budget on bench players who score nothing.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pulp

SQUAD_SPLIT = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3


@dataclass
class Squad:
    player_ids: list[int]
    starting_ids: list[int]
    total_cost: float
    xp: float


def optimize_squad(xp_df: pd.DataFrame, cfg, xp_col: str = "xp_next5",
                   must_include: list[int] | None = None,
                   banned: list[int] | None = None) -> Squad:
    pool = xp_df[~xp_df["player_id"].isin(banned or [])].reset_index(drop=True)
    ids = [int(i) for i in pool["player_id"]]
    xp = dict(zip(ids, pool[xp_col].astype(float)))
    price = dict(zip(ids, pool["price"].astype(float)))
    pos = dict(zip(ids, pool["position"]))
    club = dict(zip(ids, pool["team"]))
    bench_w = float(np.mean(cfg.bench_weight))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")

    prob += pulp.lpSum(
        xp[i] * start[i] + bench_w * xp[i] * (squad[i] - start[i]) for i in ids
    )

    prob += pulp.lpSum(price[i] * squad[i] for i in ids) <= cfg.budget
    prob += pulp.lpSum(squad[i] for i in ids) == sum(SQUAD_SPLIT.values())
    prob += pulp.lpSum(start[i] for i in ids) == XI_SIZE
    for p, n in SQUAD_SPLIT.items():
        prob += pulp.lpSum(squad[i] for i in ids if pos[i] == p) == n
    for p in SQUAD_SPLIT:
        in_pos = [start[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(in_pos) >= XI_MIN[p]
        prob += pulp.lpSum(in_pos) <= XI_MAX[p]
    for c in set(club.values()):
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= MAX_PER_CLUB
    for i in ids:
        prob += start[i] <= squad[i]
    for i in (must_include or []):
        if i not in squad:
            raise ValueError(f"must_include player {i} is not in the pool")
        prob += squad[i] == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"squad selection infeasible under the given constraints "
            f"(budget £{cfg.budget}m, status={pulp.LpStatus[status]})"
        )

    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    return Squad(
        player_ids=chosen,
        starting_ids=starters,
        total_cost=round(sum(price[i] for i in chosen), 1),
        xp=round(sum(xp[i] for i in starters), 3),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_squad.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/optimize/ tests/test_squad.py
git commit -m "feat: Mode 1 joint MILP squad builder with hard FPL constraints"
```

---

### Task 13: Lineup, bench order, captain and vice

**Files:**
- Create: `fpl/optimize/lineup.py`, `tests/test_lineup.py`

**Interfaces:**
- Consumes: `Squad` (Task 12), contract frame (Task 9)
- Produces: `@dataclass Lineup(xi: list[int], bench: list[int], formation: str, captain: int, vice: int, xp: float)` and `build_lineup(squad: Squad, xp_df: pd.DataFrame, xp_col: str = "xp_next1") -> Lineup`

**Rules:** bench ordered by descending xP, but the reserve goalkeeper always occupies bench slot 1 (an outfield sub can replace any outfield player; the spare GK cannot). Captain = highest-xP starter; vice = second-highest.

- [ ] **Step 1: Write the failing test**

`tests/test_lineup.py`:
```python
import pandas as pd
from fpl.optimize.squad import Squad
from fpl.optimize.lineup import build_lineup, Lineup

XP = pd.DataFrame({
    "player_id": list(range(1, 16)),
    "web_name": [f"P{i}" for i in range(1, 16)],
    "team": ["T"] * 15,
    "position": (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3),
    "price": [5.0] * 15,
    "xp_next1": [6.0, 1.0,            # GKs: 1 starts, 2 benched
                 5.0, 4.9, 4.8, 4.7, 0.5,   # DEFs: last is weakest
                 9.0, 8.0, 7.0, 6.5, 0.4,   # MIDs
                 8.5, 7.5, 0.3],            # FWDs
    "xp_next5": [30.0] * 15,
    "p_start": [0.9] * 15,
    "e_minutes": [80.0] * 15,
    "confidence": ["high"] * 15,
    "flags": [[] for _ in range(15)],
})
SQUAD = Squad(
    player_ids=list(range(1, 16)),
    starting_ids=[1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
    total_cost=75.0,
    xp=0.0,
)


def test_returns_eleven_starters_and_four_bench():
    lu = build_lineup(SQUAD, XP)
    assert isinstance(lu, Lineup)
    assert len(lu.xi) == 11
    assert len(lu.bench) == 4


def test_bench_and_xi_partition_the_squad():
    lu = build_lineup(SQUAD, XP)
    assert set(lu.xi) | set(lu.bench) == set(SQUAD.player_ids)
    assert not set(lu.xi) & set(lu.bench)


def test_reserve_keeper_is_first_on_the_bench():
    lu = build_lineup(SQUAD, XP)
    assert lu.bench[0] == 2  # the non-starting GKP


def test_outfield_bench_ordered_by_descending_xp():
    lu = build_lineup(SQUAD, XP)
    outfield = lu.bench[1:]
    xp = XP.set_index("player_id")["xp_next1"]
    assert list(xp.loc[outfield]) == sorted(xp.loc[outfield], reverse=True)


def test_formation_string_matches_starters():
    lu = build_lineup(SQUAD, XP)
    assert lu.formation == "4-4-2"


def test_captain_is_highest_xp_starter():
    lu = build_lineup(SQUAD, XP)
    assert lu.captain == 8  # xp 9.0


def test_vice_is_second_highest_and_differs_from_captain():
    lu = build_lineup(SQUAD, XP)
    assert lu.vice == 13  # xp 8.5
    assert lu.vice != lu.captain


def test_lineup_xp_counts_captain_twice():
    lu = build_lineup(SQUAD, XP)
    xp = XP.set_index("player_id")["xp_next1"]
    expected = sum(xp.loc[lu.xi]) + xp.loc[lu.captain]
    assert abs(lu.xp - expected) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lineup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.optimize.lineup'`

- [ ] **Step 3: Write minimal implementation**

`fpl/optimize/lineup.py`:
```python
"""Starting XI presentation: formation, bench order, captain and vice."""
from dataclasses import dataclass
import pandas as pd

from .squad import Squad


@dataclass
class Lineup:
    xi: list[int]
    bench: list[int]
    formation: str
    captain: int
    vice: int
    xp: float


def build_lineup(squad: Squad, xp_df: pd.DataFrame, xp_col: str = "xp_next1") -> Lineup:
    df = xp_df.set_index("player_id")
    xi = list(squad.starting_ids)
    bench_ids = [i for i in squad.player_ids if i not in set(xi)]

    # The reserve keeper can only replace the keeper, so it always sits in slot 1.
    keepers = [i for i in bench_ids if df.loc[i, "position"] == "GKP"]
    outfield = [i for i in bench_ids if df.loc[i, "position"] != "GKP"]
    outfield.sort(key=lambda i: float(df.loc[i, xp_col]), reverse=True)
    bench = keepers + outfield

    counts = df.loc[xi, "position"].value_counts()
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"

    ranked = sorted(xi, key=lambda i: float(df.loc[i, xp_col]), reverse=True)
    captain, vice = ranked[0], ranked[1]
    total = sum(float(df.loc[i, xp_col]) for i in xi) + float(df.loc[captain, xp_col])

    return Lineup(xi=xi, bench=bench, formation=formation,
                  captain=captain, vice=vice, xp=round(total, 3))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lineup.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/optimize/lineup.py tests/test_lineup.py
git commit -m "feat: lineup, bench ordering, captain and vice selection"
```

---

### Task 14: Free-transfer state tracking

**Files:**
- Create: `fpl/state.py`, `tests/test_state.py`

**Interfaces:**
- Consumes: `Config` (Task 0)
- Produces: `@dataclass State(free_transfers: int, last_event: int, chips_used: list[str])`, `load_state(path: Path, cfg: Config) -> State`, `save_state(state: State, path: Path) -> None`, `advance_ft(state: State, transfers_made: int, chip: str | None = None) -> int`, `reconcile(state: State, entry_history: dict) -> tuple[int, bool]` returning `(free_transfers, matched)`. Constants `FT_CAP = 5`, `CHIPS_PRESERVING_FT = {"wildcard", "freehit"}`.

**Why this module exists:** the FT count is not exposed by any public endpoint (the only source is auth-gated and off-limits per Global Constraints), so it must be tracked locally.

**Correction to spec §8's simplified formula.** The spec writes `FT_next = min(5, FT_current + 1 − transfers_made)`, which goes wrong when hits are taken: with 2 FT and 3 transfers made it yields 0, but the true answer is 1 (the balance floors at 0, *then* accrues +1). Implement `min(5, max(0, FT − used) + 1)`.

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
import json
from pathlib import Path
from fpl.config import Config
from fpl.state import (
    State, load_state, save_state, advance_ft, reconcile, FT_CAP,
)


def test_load_state_seeds_from_config_when_file_absent(tmp_path):
    s = load_state(tmp_path / "state.json", Config(free_transfers=2))
    assert s.free_transfers == 2
    assert s.chips_used == []


def test_save_then_load_roundtrips(tmp_path):
    p = tmp_path / "state.json"
    save_state(State(free_transfers=3, last_event=5, chips_used=["wildcard"]), p)
    s = load_state(p, Config(free_transfers=1))
    assert s.free_transfers == 3
    assert s.last_event == 5
    assert s.chips_used == ["wildcard"]


def test_unused_transfer_accrues():
    assert advance_ft(State(1, 1, []), transfers_made=0) == 2


def test_ft_caps_at_five():
    assert advance_ft(State(FT_CAP, 1, []), transfers_made=0) == FT_CAP
    assert advance_ft(State(FT_CAP, 1, []), transfers_made=1) == FT_CAP


def test_using_the_free_transfer_returns_to_one():
    assert advance_ft(State(1, 1, []), transfers_made=1) == 1


def test_taking_hits_floors_the_balance_at_zero_before_accruing():
    # 2 FT, 3 transfers made (one -4 hit) -> balance 0, then +1
    assert advance_ft(State(2, 1, []), transfers_made=3) == 1


def test_wildcard_preserves_the_balance_and_still_accrues():
    assert advance_ft(State(3, 1, []), transfers_made=9, chip="wildcard") == 4


def test_free_hit_preserves_the_balance():
    assert advance_ft(State(2, 1, []), transfers_made=11, chip="freehit") == 3


def test_bench_boost_does_not_preserve_the_balance():
    assert advance_ft(State(2, 1, []), transfers_made=2, chip="benchboost") == 1


def test_reconcile_matches_when_history_agrees():
    state = State(free_transfers=2, last_event=3, chips_used=[])
    history = {"current": [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 1},
    ]}
    ft, matched = reconcile(state, history)
    assert matched is True
    assert ft == 2


def test_reconcile_reports_drift_and_prefers_derived_value():
    state = State(free_transfers=5, last_event=3, chips_used=[])
    history = {"current": [
        {"event": 1, "event_transfers": 1},
        {"event": 2, "event_transfers": 1},
        {"event": 3, "event_transfers": 1},
    ]}
    ft, matched = reconcile(state, history)
    assert matched is False
    assert ft == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.state'`

- [ ] **Step 3: Write minimal implementation**

`fpl/state.py`:
```python
"""Persisted free-transfer balance and chip usage.

The FT count is not available from any public endpoint - the only source is
auth-gated and off-limits - so it is tracked locally and reconciled against
entry/{id}/history/ event_transfers.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

FT_CAP = 5
CHIPS_PRESERVING_FT = {"wildcard", "freehit"}


@dataclass
class State:
    free_transfers: int = 1
    last_event: int = 0
    chips_used: list[str] = field(default_factory=list)


def load_state(path: Path, cfg) -> State:
    p = Path(path)
    if not p.exists():
        return State(free_transfers=cfg.free_transfers, last_event=0, chips_used=[])
    raw = json.loads(p.read_text())
    return State(
        free_transfers=int(raw.get("free_transfers", cfg.free_transfers)),
        last_event=int(raw.get("last_event", 0)),
        chips_used=list(raw.get("chips_used", [])),
    )


def save_state(state: State, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))


def advance_ft(state: State, transfers_made: int, chip: str | None = None) -> int:
    """Balance floors at 0 before the weekly +1 accrues. Chips preserve the balance."""
    used = 0 if chip in CHIPS_PRESERVING_FT else int(transfers_made)
    return min(FT_CAP, max(0, int(state.free_transfers) - used) + 1)


def reconcile(state: State, entry_history: dict) -> tuple[int, bool]:
    """Derive the FT balance from the API's per-event transfer counts."""
    derived = State(free_transfers=1, last_event=0, chips_used=[])
    for event in entry_history.get("current", []):
        derived.free_transfers = advance_ft(derived, int(event.get("event_transfers", 0)))
        derived.last_event = int(event.get("event", derived.last_event))
    matched = derived.free_transfers == state.free_transfers
    return derived.free_transfers, matched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/state.py tests/test_state.py
git commit -m "feat: free-transfer state tracking with reconciliation"
```

---

### Task 15: Mode 2 — weekly transfer optimization

**Files:**
- Create: `fpl/optimize/transfers.py`, `tests/test_transfers.py`

**Interfaces:**
- Consumes: contract frame (Task 9), `optimize_squad` (Task 12), `Config` (Task 0)
- Produces: `@dataclass TransferPlan(out_ids, in_ids, n_transfers, hit_cost, squad_ids, starting_ids, gross_xp, net_xp, baseline_xp, gain)` and `optimize_transfers(xp_df, current_squad_ids, bank, free_transfers, cfg, xp_col="xp_next5") -> tuple[TransferPlan, list[TransferPlan]]` returning `(best, all_options)` where `all_options[0]` is always the 0-transfer baseline.

**Search depth (Global Constraints):** `range(0, free_transfers + cfg.max_paid_hits + 1)` — **not a fixed 3**. With 5 banked FTs a fixed cap would discard free moves. Budget available = `bank + Σ price(sold)`.

**Dead code until a team ID exists** — it must not block Mode 1.

- [ ] **Step 1: Write the failing test**

`tests/test_transfers.py`:
```python
import pandas as pd
from fpl.config import Config
from fpl.optimize.transfers import optimize_transfers, TransferPlan


def make_pool():
    rows, pid = [], 1
    for t in range(8):
        for pos, n in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(n):
                rows.append({
                    "player_id": pid, "web_name": f"P{pid}", "team": f"T{t}",
                    "position": pos, "price": 5.0, "xp_next1": 2.0,
                    "xp_next5": 10.0, "p_start": 0.9, "e_minutes": 80.0,
                    "confidence": "high", "flags": [],
                })
                pid += 1
    return pd.DataFrame(rows)


POOL = make_pool()
# A legal starting squad: 2 GKP, 5 DEF, 5 MID, 3 FWD across 5 clubs (3 max each)
CURRENT = (
    list(POOL[POOL.position == "GKP"].player_id[:2])
    + list(POOL[POOL.position == "DEF"].player_id[:5])
    + list(POOL[POOL.position == "MID"].player_id[:5])
    + list(POOL[POOL.position == "FWD"].player_id[:3])
)


def _pool_with_star(star_xp=60.0):
    pool = POOL.copy()
    # a cheap, uncaptured superstar in a club not already at the limit
    target = pool[(~pool.player_id.isin(CURRENT)) & (pool.position == "MID")].iloc[0]
    pool.loc[pool.player_id == target.player_id, "xp_next5"] = star_xp
    return pool, int(target.player_id)


def test_baseline_is_always_reported_first():
    cfg = Config(max_paid_hits=1)
    best, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert options[0].n_transfers == 0
    assert options[0].hit_cost == 0
    assert isinstance(best, TransferPlan)


def test_no_transfer_when_squad_already_optimal():
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert best.n_transfers == 0
    assert best.out_ids == [] and best.in_ids == []


def test_takes_a_free_transfer_for_a_clear_upgrade():
    pool, star = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert best.n_transfers == 1
    assert star in best.in_ids


def test_hit_cost_applied_beyond_free_allowance():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=2, hit_cost=4)
    _, options = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    two = next(o for o in options if o.n_transfers == 2)
    assert two.hit_cost == 4
    assert abs(two.net_xp - (two.gross_xp - 4)) < 1e-6


def test_search_depth_scales_with_banked_free_transfers():
    cfg = Config(max_paid_hits=1)
    _, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=5, cfg=cfg)
    assert max(o.n_transfers for o in options) == 6  # 5 free + 1 paid


def test_free_transfers_incur_no_hit():
    cfg = Config(max_paid_hits=0)
    _, options = optimize_transfers(POOL, CURRENT, bank=0.0, free_transfers=3, cfg=cfg)
    assert all(o.hit_cost == 0 for o in options)


def test_gain_is_relative_to_baseline():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, options = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert abs(best.gain - (best.net_xp - options[0].net_xp)) < 1e-6
    assert best.gain >= 0


def test_result_squad_respects_all_constraints():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    picked = pool[pool.player_id.isin(best.squad_ids)]
    assert len(best.squad_ids) == 15
    assert picked.team.value_counts().max() <= 3
    counts = picked.position.value_counts().to_dict()
    assert counts["GKP"] == 2 and counts["DEF"] == 5
    assert counts["MID"] == 5 and counts["FWD"] == 3


def test_transfer_counts_are_balanced():
    pool, _ = _pool_with_star()
    cfg = Config(max_paid_hits=1)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.0, free_transfers=1, cfg=cfg)
    assert len(best.out_ids) == len(best.in_ids) == best.n_transfers


def test_budget_respects_bank_plus_sale_proceeds():
    pool = POOL.copy()
    pool["price"] = 5.0
    expensive = pool[~pool.player_id.isin(CURRENT)].iloc[0].player_id
    pool.loc[pool.player_id == expensive, ["price", "xp_next5"]] = [9.0, 99.0]
    cfg = Config(max_paid_hits=0, budget=75.0)
    best, _ = optimize_transfers(pool, CURRENT, bank=0.5, free_transfers=1, cfg=cfg)
    cost = pool[pool.player_id.isin(best.squad_ids)]["price"].sum()
    assert cost <= 15 * 5.0 + 0.5 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transfers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.optimize.transfers'`

- [ ] **Step 3: Write minimal implementation**

`fpl/optimize/transfers.py`:
```python
"""Mode 2: weekly transfer optimization.

Search depth is free_transfers + max_paid_hits, never a fixed constant:
free transfers bank up to 5, and a fixed cap would silently discard moves
that cost nothing.
"""
from dataclasses import dataclass, field
import pandas as pd
import pulp

from .squad import SQUAD_SPLIT, XI_MIN, XI_MAX, XI_SIZE, MAX_PER_CLUB
import numpy as np


@dataclass
class TransferPlan:
    out_ids: list[int] = field(default_factory=list)
    in_ids: list[int] = field(default_factory=list)
    n_transfers: int = 0
    hit_cost: int = 0
    squad_ids: list[int] = field(default_factory=list)
    starting_ids: list[int] = field(default_factory=list)
    gross_xp: float = 0.0
    net_xp: float = 0.0
    baseline_xp: float = 0.0
    gain: float = 0.0


def _solve(xp_df, current, budget, max_changes, cfg, xp_col):
    ids = [int(i) for i in xp_df["player_id"]]
    xp = dict(zip(ids, xp_df[xp_col].astype(float)))
    price = dict(zip(ids, xp_df["price"].astype(float)))
    pos = dict(zip(ids, xp_df["position"]))
    club = dict(zip(ids, xp_df["team"]))
    bench_w = float(np.mean(cfg.bench_weight))
    current_set = set(int(i) for i in current)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    prob += pulp.lpSum(
        xp[i] * start[i] + bench_w * xp[i] * (squad[i] - start[i]) for i in ids
    )
    prob += pulp.lpSum(price[i] * squad[i] for i in ids) <= budget
    prob += pulp.lpSum(squad[i] for i in ids) == sum(SQUAD_SPLIT.values())
    prob += pulp.lpSum(start[i] for i in ids) == XI_SIZE
    for p, n in SQUAD_SPLIT.items():
        prob += pulp.lpSum(squad[i] for i in ids if pos[i] == p) == n
        in_pos = [start[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(in_pos) >= XI_MIN[p]
        prob += pulp.lpSum(in_pos) <= XI_MAX[p]
    for c in set(club.values()):
        prob += pulp.lpSum(squad[i] for i in ids if club[i] == c) <= MAX_PER_CLUB
    for i in ids:
        prob += start[i] <= squad[i]
    # keep at least 15 - max_changes of the current squad
    prob += pulp.lpSum(squad[i] for i in ids if i in current_set) >= 15 - max_changes

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
        return None
    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    return chosen, starters, sum(xp[i] for i in starters)


def optimize_transfers(xp_df: pd.DataFrame, current_squad_ids: list[int], bank: float,
                       free_transfers: int, cfg, xp_col: str = "xp_next5"):
    current_set = set(int(i) for i in current_squad_ids)
    price = dict(zip(xp_df["player_id"].astype(int), xp_df["price"].astype(float)))
    budget = float(bank) + sum(price[i] for i in current_set)

    options: list[TransferPlan] = []
    for n in range(0, int(free_transfers) + int(cfg.max_paid_hits) + 1):
        solved = _solve(xp_df, current_set, budget, n, cfg, xp_col)
        if solved is None:
            continue
        chosen, starters, gross = solved
        actual = len(current_set - set(chosen))
        paid = max(0, actual - int(free_transfers))
        hit = paid * int(cfg.hit_cost)
        options.append(TransferPlan(
            out_ids=sorted(current_set - set(chosen)),
            in_ids=sorted(set(chosen) - current_set),
            n_transfers=actual,
            hit_cost=hit,
            squad_ids=chosen,
            starting_ids=starters,
            gross_xp=round(gross, 3),
            net_xp=round(gross - hit, 3),
        ))

    # options[0] is the 0-transfer baseline; always keep it visible
    baseline = options[0].net_xp if options else 0.0
    for o in options:
        o.baseline_xp = baseline
        o.gain = round(o.net_xp - baseline, 3)
    best = max(options, key=lambda o: o.net_xp)
    return best, options
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transfers.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/optimize/transfers.py tests/test_transfers.py
git commit -m "feat: Mode 2 transfer optimization with FT-aware search depth"
```

---

### Task 16: Chip advisor

**Files:**
- Create: `fpl/optimize/chips.py`, `tests/test_chips.py`

**Interfaces:**
- Consumes: contract frame (Task 9), `Lineup` (Task 13), `fixture_counts` (Task 5)
- Produces: `@dataclass ChipAdvice(chip: str | None, reason: str)` and `advise_chips(xp_df, lineup, squad_ids, counts, team_by_player: dict[int, int], from_event: int, chips_used: list[str]) -> ChipAdvice`

**Heuristic, not optimized (spec §8).** Never auto-recommend without stating the tradeoff. Rules: **Bench Boost** only when every bench player's `xp_next1` clears `BENCH_BOOST_MIN_XP`; **Triple Captain** when the captain has a double gameweek or an outstanding single fixture; **Free Hit** when 3+ squad players blank; **Wildcard** when 4+ squad players are flagged unavailable or doubtful. Already-used chips are never suggested.

- [ ] **Step 1: Write the failing test**

`tests/test_chips.py`:
```python
import pandas as pd
from fpl.optimize.lineup import Lineup
from fpl.optimize.chips import advise_chips, ChipAdvice, BENCH_BOOST_MIN_XP

SQUAD = list(range(1, 16))
TEAM_BY_PLAYER = {i: 1 for i in SQUAD}
LINEUP = Lineup(xi=list(range(1, 12)), bench=[12, 13, 14, 15],
                formation="4-4-2", captain=1, vice=2, xp=60.0)


def _xp(bench_xp=1.0, captain_xp=8.0, flags=None):
    return pd.DataFrame({
        "player_id": SQUAD,
        "web_name": [f"P{i}" for i in SQUAD],
        "team": ["T1"] * 15,
        "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        "price": [5.0] * 15,
        "xp_next1": [captain_xp] + [4.0] * 10 + [bench_xp] * 4,
        "xp_next5": [30.0] * 15,
        "p_start": [0.9] * 15,
        "e_minutes": [80.0] * 15,
        "confidence": ["high"] * 15,
        "flags": flags or [[] for _ in SQUAD],
    })


def _counts(n=1):
    return pd.DataFrame([{"team_id": 1, "event": 1, "n_fixtures": n}])


def test_no_chip_recommended_in_a_normal_week():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, 1, [])
    assert isinstance(a, ChipAdvice)
    assert a.chip is None
    assert a.reason


def test_bench_boost_when_bench_is_strong():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, [])
    assert a.chip == "benchboost"


def test_no_bench_boost_when_one_bench_player_is_weak():
    xp = _xp(bench_xp=BENCH_BOOST_MIN_XP + 1)
    xp.loc[xp.player_id == 15, "xp_next1"] = 0.1
    a = advise_chips(xp, LINEUP, SQUAD, _counts(), TEAM_BY_PLAYER, 1, [])
    assert a.chip != "benchboost"


def test_triple_captain_on_a_double_gameweek():
    a = advise_chips(_xp(captain_xp=12.0), LINEUP, SQUAD, _counts(n=2),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip == "triplecaptain"


def test_free_hit_when_several_players_blank():
    a = advise_chips(_xp(), LINEUP, SQUAD, _counts(n=0), TEAM_BY_PLAYER, 1, [])
    assert a.chip == "freehit"


def test_wildcard_when_squad_riddled_with_problems():
    flags = [["Unavailable (i): injured"]] * 5 + [[] for _ in range(10)]
    a = advise_chips(_xp(flags=flags), LINEUP, SQUAD, _counts(),
                     TEAM_BY_PLAYER, 1, [])
    assert a.chip == "wildcard"


def test_used_chips_are_never_suggested_again():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, ["benchboost"])
    assert a.chip != "benchboost"


def test_reason_always_explains_the_tradeoff():
    a = advise_chips(_xp(bench_xp=BENCH_BOOST_MIN_XP + 1), LINEUP, SQUAD,
                     _counts(), TEAM_BY_PLAYER, 1, [])
    assert len(a.reason) > 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.optimize.chips'`

- [ ] **Step 3: Write minimal implementation**

`fpl/optimize/chips.py`:
```python
"""Chip advisor - heuristic flags, never an unexplained auto-recommendation."""
from dataclasses import dataclass
import pandas as pd

BENCH_BOOST_MIN_XP = 2.5
TRIPLE_CAPTAIN_MIN_XP = 9.0
FREE_HIT_MIN_BLANKS = 3
WILDCARD_MIN_PROBLEMS = 4
PROBLEM_MARKERS = ("Unavailable", "Doubtful")


@dataclass
class ChipAdvice:
    chip: str | None
    reason: str


def advise_chips(xp_df: pd.DataFrame, lineup, squad_ids: list[int],
                 counts: pd.DataFrame, team_by_player: dict[int, int],
                 from_event: int, chips_used: list[str]) -> ChipAdvice:
    df = xp_df.set_index("player_id")
    used = set(chips_used or [])
    n_by_team = {
        int(r["team_id"]): int(r["n_fixtures"])
        for _, r in counts[counts["event"] == from_event].iterrows()
    }

    def fixtures_for(pid: int) -> int:
        return n_by_team.get(team_by_player.get(int(pid), -1), 1)

    blanks = sum(1 for pid in squad_ids if fixtures_for(pid) == 0)
    problems = sum(
        1 for pid in squad_ids
        if any(m in f for f in df.loc[pid, "flags"] for m in PROBLEM_MARKERS)
    )

    if "freehit" not in used and blanks >= FREE_HIT_MIN_BLANKS:
        return ChipAdvice("freehit", (
            f"{blanks} of your 15 have no fixture this gameweek. A Free Hit fields a "
            f"one-week replacement squad, but you lose it for a future blank or double "
            f"— only worth it if you can't cover the gap with transfers."
        ))

    if "wildcard" not in used and problems >= WILDCARD_MIN_PROBLEMS:
        return ChipAdvice("wildcard", (
            f"{problems} players carry injury or rotation flags. A Wildcard fixes them "
            f"all at once with unlimited free transfers, but spends a chip you may want "
            f"later for a fixture swing."
        ))

    cap_xp = float(df.loc[lineup.captain, "xp_next1"])
    cap_fixtures = fixtures_for(lineup.captain)
    if "triplecaptain" not in used and (cap_fixtures >= 2 or cap_xp >= TRIPLE_CAPTAIN_MIN_XP):
        detail = "a double gameweek" if cap_fixtures >= 2 else "an outstanding single fixture"
        return ChipAdvice("triplecaptain", (
            f"{df.loc[lineup.captain, 'web_name']} has {detail} (xP {cap_xp:.1f}). "
            f"Triple Captain turns that into 3x, but a blank or an early substitution "
            f"wastes the chip entirely."
        ))

    bench_xp = [float(df.loc[pid, "xp_next1"]) for pid in lineup.bench]
    if "benchboost" not in used and bench_xp and min(bench_xp) >= BENCH_BOOST_MIN_XP:
        return ChipAdvice("benchboost", (
            f"All four bench players project at {min(bench_xp):.1f}+ xP "
            f"({sum(bench_xp):.1f} total). Bench Boost banks that, but a stronger bench "
            f"week may come later — especially in a double gameweek."
        ))

    return ChipAdvice(None, (
        "No chip recommended this week — the bench is too weak for a Bench Boost, "
        "no standout double-gameweek captain, and the squad has no cluster of blanks "
        "or injuries needing a Wildcard or Free Hit."
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chips.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/optimize/chips.py tests/test_chips.py
git commit -m "feat: chip advisor with explained tradeoffs"
```

---

### Task 17: Weekly report

**Files:**
- Create: `fpl/report/__init__.py`, `fpl/report/weekly.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `Lineup` (Task 13), `ChipAdvice` (Task 16), `TransferPlan` (Task 15), contract frame (Task 9)
- Produces: `@dataclass Recommendation(gw, deadline, mode, lineup, squad_ids, transfers, chip, bank, squad_value, flags, stale, trust)` and `render(rec: Recommendation, xp_df: pd.DataFrame) -> str`

**Format is the `weekly-report` skill's, verbatim** — sections in this order: `## Gameweek {N} — {deadline}`, `### Starting XI ({formation})`, `### Bench (in order)`, `### Captain / Vice`, `### Transfers this week`, `### Chip watch`, `### Budget`, `### Flags`.

**Rules enforced by tests:** recommendation phrasing only (never "transferred"/"captained"); one line of "why" per major decision; every flag on a selected player surfaces; uncertainty stated explicitly.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import pandas as pd
from fpl.optimize.lineup import Lineup
from fpl.optimize.chips import ChipAdvice
from fpl.optimize.transfers import TransferPlan
from fpl.report.weekly import render, Recommendation

XP = pd.DataFrame({
    "player_id": list(range(1, 16)),
    "web_name": [f"Player{i}" for i in range(1, 16)],
    "team": ["Arsenal"] * 15,
    "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
    "price": [5.0] * 15,
    "xp_next1": [4.0] * 15,
    "xp_next5": [20.0] * 15,
    "p_start": [0.9] * 15,
    "e_minutes": [80.0] * 15,
    "confidence": ["high"] * 13 + ["low"] * 2,
    "flags": [[] for _ in range(13)] + [["Doubtful: 25% chance of playing"],
                                        ["Limited data: no Premier League minutes"]],
})
LINEUP = Lineup(xi=[1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14], bench=[2, 7, 12, 15],
                formation="4-4-2", captain=8, vice=13, xp=52.0)


def _rec(**kw):
    base = dict(
        gw=1, deadline="Fri 21 Aug 2026, 18:30 UK", mode=1, lineup=LINEUP,
        squad_ids=list(range(1, 16)), transfers=None,
        chip=ChipAdvice(None, "No chip recommended this week — bench too weak."),
        bank=0.5, squad_value=99.5, flags=[], stale=False,
        trust="Model beats both baselines in every position.",
    )
    base.update(kw)
    return Recommendation(**base)


def test_contains_all_required_sections():
    out = render(_rec(), XP)
    for section in ["## Gameweek 1", "### Starting XI", "### Bench (in order)",
                    "### Captain", "### Transfers this week", "### Chip watch",
                    "### Budget", "### Flags"]:
        assert section in out


def test_formation_and_deadline_in_header():
    out = render(_rec(), XP)
    assert "4-4-2" in out
    assert "Fri 21 Aug 2026, 18:30 UK" in out


def test_bench_is_numbered_in_order():
    out = render(_rec(), XP)
    bench_block = out.split("### Bench (in order)")[1].split("###")[0]
    for n in range(1, 5):
        assert f"{n}." in bench_block


def test_captain_and_vice_named_with_reasoning():
    out = render(_rec(), XP)
    block = out.split("### Captain")[1].split("###")[0]
    assert "Player8" in block and "Player13" in block
    assert len(block.strip().splitlines()) >= 2  # names plus a why line


def test_no_transfer_message_when_none_recommended():
    out = render(_rec(), XP)
    assert "No transfer recommended" in out


def test_transfer_rendered_with_net_gain():
    plan = TransferPlan(out_ids=[3], in_ids=[4], n_transfers=1, hit_cost=0,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=60.0, baseline_xp=57.5, gain=2.5)
    out = render(_rec(transfers=plan), XP)
    assert "Player3" in out and "Player4" in out
    assert "2.5" in out


def test_never_uses_past_tense_action_verbs():
    plan = TransferPlan(out_ids=[3], in_ids=[4], n_transfers=1, hit_cost=4,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=56.0, baseline_xp=55.0, gain=1.0)
    out = render(_rec(transfers=plan), XP).lower()
    for banned in ["transferred", "captained", "benched him", "i have made"]:
        assert banned not in out


def test_uses_recommendation_language():
    out = render(_rec(), XP).lower()
    assert "recommend" in out or "suggest" in out


def test_flags_on_squad_players_are_surfaced():
    out = render(_rec(), XP)
    assert "Doubtful: 25% chance of playing" in out
    assert "Limited data" in out


def test_stale_data_warning_shown():
    out = render(_rec(stale=True), XP)
    assert "stale" in out.lower()


def test_trust_summary_included():
    out = render(_rec(trust="LOW CONFIDENCE — DEF failed validation"), XP)
    assert "LOW CONFIDENCE" in out


def test_hit_cost_disclosed_when_taken():
    plan = TransferPlan(out_ids=[3, 5], in_ids=[4, 6], n_transfers=2, hit_cost=4,
                        squad_ids=list(range(1, 16)), starting_ids=LINEUP.xi,
                        gross_xp=60.0, net_xp=56.0, baseline_xp=55.0, gain=1.0)
    out = render(_rec(transfers=plan), XP)
    assert "-4" in out or "−4" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.report'`

- [ ] **Step 3: Write minimal implementation**

`fpl/report/__init__.py`: empty file.

`fpl/report/weekly.py`:
```python
"""Final user-facing gameweek report.

Presentation only - never recomputes anything. Phrasing is always a
recommendation for the user to apply manually in the FPL app.
"""
from dataclasses import dataclass, field
import pandas as pd

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]


@dataclass
class Recommendation:
    gw: int
    deadline: str
    mode: int
    lineup: object
    squad_ids: list[int]
    transfers: object | None = None
    chip: object | None = None
    bank: float = 0.0
    squad_value: float = 0.0
    flags: list[str] = field(default_factory=list)
    stale: bool = False
    trust: str = ""


def _name(df, pid) -> str:
    row = df.loc[pid]
    return f"{row['web_name']} ({row['team']}, £{row['price']}m, {row['xp_next1']:.1f} xP)"


def render(rec: Recommendation, xp_df: pd.DataFrame) -> str:
    df = xp_df.set_index("player_id")
    lu = rec.lineup
    out = [f"## Gameweek {rec.gw} — {rec.deadline}", ""]

    if rec.stale:
        out += ["> ⚠️ Live data was unavailable — this uses the most recent cached "
                "snapshot and may be stale.", ""]

    out += [f"### Starting XI ({lu.formation})"]
    for pos in POSITION_ORDER:
        members = [p for p in lu.xi if df.loc[p, "position"] == pos]
        if members:
            members.sort(key=lambda p: float(df.loc[p, "xp_next1"]), reverse=True)
            out.append(f"{pos}: " + ", ".join(_name(df, p) for p in members))
    out.append("")

    out.append("### Bench (in order)")
    for n, pid in enumerate(lu.bench, start=1):
        note = " — reserve keeper" if df.loc[pid, "position"] == "GKP" else ""
        out.append(f"{n}. {_name(df, pid)}{note}")
    out.append("")

    cap, vice = df.loc[lu.captain], df.loc[lu.vice]
    out += [
        f"### Captain: {cap['web_name']} (C)  |  Vice: {vice['web_name']} (VC)",
        f"Recommended because {cap['web_name']} has the highest projected return in the "
        f"squad ({cap['xp_next1']:.1f} xP vs {vice['xp_next1']:.1f} for {vice['web_name']}).",
        "",
    ]

    out.append("### Transfers this week")
    t = rec.transfers
    if t is None or t.n_transfers == 0:
        out.append("No transfer recommended — the squad is already optimal on projected points.")
    else:
        for o, i in zip(t.out_ids, t.in_ids):
            out.append(f"{df.loc[o, 'web_name']} → {df.loc[i, 'web_name']}")
        hit = f" after a -{t.hit_cost} hit" if t.hit_cost else " (no hit — within your free transfers)"
        out.append(f"Suggested net gain of {t.gain:.1f} xP over the next gameweeks{hit}.")
    out.append("")

    out.append("### Chip watch")
    if rec.chip is None or rec.chip.chip is None:
        out.append(rec.chip.reason if rec.chip else "No chip recommended this week.")
    else:
        out.append(f"**{rec.chip.chip}** — {rec.chip.reason}")
    out.append("")

    out += [f"### Budget", f"Bank: £{rec.bank}m | Squad value: £{rec.squad_value}m", ""]

    out.append("### Flags")
    lines = []
    for pid in rec.squad_ids:
        for f in df.loc[pid, "flags"]:
            lines.append(f"- {df.loc[pid, 'web_name']}: {f}")
    lines += [f"- {f}" for f in rec.flags]
    if rec.trust:
        lines.append(f"- Model confidence: {rec.trust}")
    out += lines or ["- No outstanding injury or rotation concerns in this squad."]

    return "\n".join(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add fpl/report/ tests/test_report.py
git commit -m "feat: weekly report renderer in the skill's format"
```

---

### Task 18: Pipeline entry point and headless verification

**Files:**
- Create: `fpl/pipeline.py`, `run_gameweek.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: every module above
- Produces: `run(cfg: Config, mode: int, from_event: int, root: Path, client=None, news=None) -> tuple[Recommendation, pd.DataFrame]` and a CLI with flags `--mode {1,2}`, `--gw N`, `--no-refresh`, `--deep`, `--backtest`

**Purity requirement (spec §11):** `run_gameweek.py` and `fpl/pipeline.py` must import **no MCP client and no skill machinery** — only the HTTP client and cached data. A test asserts this, so the cron path works by construction rather than by hope. `fpl.data.archive` must also stay out of the weekly path (backtest only).

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from pathlib import Path
import pandas as pd
from fpl.config import Config
from fpl.pipeline import run
from fpl.report.weekly import Recommendation

BOOTSTRAP = {
    "teams": [
        {"id": t, "name": f"Team{t}", "short_name": f"T{t}",
         "strength_overall_home": 3, "strength_overall_away": 3}
        for t in range(1, 9)
    ],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ],
    "elements": [],
}
_pid = 1
for _t in range(1, 9):
    for _et, _n in [(1, 3), (2, 6), (3, 6), (4, 4)]:
        for _k in range(_n):
            BOOTSTRAP["elements"].append({
                "id": _pid, "web_name": f"P{_pid}", "team": _t, "element_type": _et,
                "now_cost": 45 + (_pid % 5) * 10, "status": "a", "news": "",
                "chance_of_playing_next_round": None, "minutes": 2500, "starts": 28,
                "total_points": 100 + _pid % 40, "goals_scored": _pid % 8,
                "assists": _pid % 5, "clean_sheets": 10, "goals_conceded": 30,
                "saves": 60 if _et == 1 else 0, "bonus": 10, "bps": 400,
                "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
                "expected_goals": str(_pid % 8), "expected_assists": str(_pid % 5),
                "expected_goals_conceded": "30.0", "selected_by_percent": "5.0",
            })
            _pid += 1

FIXTURES = []
_fid = 1
for _ev in range(1, 7):
    for _h, _a in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        FIXTURES.append({
            "id": _fid, "event": _ev, "team_h": _h, "team_a": _a,
            "team_h_difficulty": 3, "team_a_difficulty": 3,
            "kickoff_time": f"2026-08-2{_ev}T19:00:00Z", "finished": False,
        })
        _fid += 1


class FakeClient:
    stale = False

    def bootstrap(self):
        return BOOTSTRAP

    def fixtures(self):
        return FIXTURES


def test_mode_one_produces_a_valid_recommendation(tmp_path):
    rec, xp = run(Config(budget=100.0), mode=1, from_event=1, root=tmp_path,
                  client=FakeClient())
    assert isinstance(rec, Recommendation)
    assert len(rec.squad_ids) == 15
    assert len(rec.lineup.xi) == 11
    assert len(rec.lineup.bench) == 4
    assert rec.lineup.captain in rec.lineup.xi


def test_mode_one_respects_budget(tmp_path):
    rec, _ = run(Config(budget=100.0), mode=1, from_event=1, root=tmp_path,
                 client=FakeClient())
    assert rec.squad_value <= 100.0 + 1e-6


def test_returns_contract_frame(tmp_path):
    from fpl.model.xp import CONTRACT_COLUMNS
    _, xp = run(Config(), mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert list(xp.columns) == CONTRACT_COLUMNS


def test_pipeline_imports_no_mcp_or_archive():
    """The headless path must not depend on MCP, skills, or the backtest archive."""
    src = Path("fpl/pipeline.py").read_text() + Path("run_gameweek.py").read_text()
    for banned in ["mcp", "fantasy_pl", "fpl_mcp", "archive", "Skill"]:
        assert banned not in src, f"weekly path must not reference {banned!r}"


def test_stale_flag_propagates_to_report(tmp_path):
    class StaleClient(FakeClient):
        stale = True

    rec, _ = run(Config(), mode=1, from_event=1, root=tmp_path, client=StaleClient())
    assert rec.stale is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fpl.pipeline'`

- [ ] **Step 3: Write minimal implementation**

`fpl/pipeline.py`:
```python
"""End-to-end weekly pipeline: data -> model -> optimize -> report.

Pure Python. No MCP, no skills, no third-party archive - so a headless cron
invocation works by construction.
"""
from pathlib import Path
import pandas as pd

from .config import Config
from .data.cache import Cache
from .data.client import FplClient
from .data.normalize import normalize_players, normalize_teams, normalize_fixtures
from .data.store import save_table
from .model.strength import team_ratings
from .model.minutes import minutes_model
from .model.scoring import per90_rates
from .model.fixtures import team_fixture_frame, fixture_counts
from .model.xp import build_xp
from .optimize.squad import optimize_squad
from .optimize.lineup import build_lineup
from .optimize.chips import advise_chips
from .optimize.transfers import optimize_transfers
from .report.weekly import Recommendation, render


def run(cfg: Config, mode: int, from_event: int, root: Path, client=None,
        news=None, current_squad=None, bank: float = 0.0, free_transfers: int = 1):
    root = Path(root)
    client = client or FplClient(Cache(root / "cache"), ttl_hours=cfg.cache_ttl_hours)

    bootstrap = client.bootstrap()
    raw_fixtures = client.fixtures()
    players = normalize_players(bootstrap)
    teams = normalize_teams(bootstrap)
    fixtures = normalize_fixtures(raw_fixtures)

    processed = root / "processed"
    save_table(players, "players", processed)
    save_table(teams, "teams", processed)
    save_table(fixtures, "fixtures", processed)

    ratings = team_ratings(players, teams)
    tfx = team_fixture_frame(fixtures, ratings, from_event, cfg.horizon_gw)
    counts = fixture_counts(fixtures, list(teams["team_id"]), from_event, cfg.horizon_gw)
    rates = per90_rates(players, cfg)
    minutes = minutes_model(players, cfg, news=news)
    xp = build_xp(players, rates, minutes, tfx, counts, cfg, from_event)

    transfers = None
    if mode == 2 and current_squad:
        best, _options = optimize_transfers(xp, current_squad, bank, free_transfers, cfg)
        squad_ids, starting_ids, transfers = best.squad_ids, best.starting_ids, best
    else:
        squad = optimize_squad(xp, cfg)
        squad_ids, starting_ids = squad.player_ids, squad.starting_ids

    from .optimize.squad import Squad
    lineup = build_lineup(Squad(squad_ids, starting_ids, 0.0, 0.0), xp)

    team_by_player = dict(zip(players["player_id"].astype(int), players["team_id"].astype(int)))
    chip = advise_chips(xp, lineup, squad_ids, counts, team_by_player, from_event, [])

    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    value = round(sum(prices[i] for i in squad_ids), 1)
    deadline = next(
        (e["deadline_time"] for e in bootstrap.get("events", []) if e["id"] == from_event),
        "see the FPL site",
    )

    rec = Recommendation(
        gw=from_event, deadline=deadline, mode=mode, lineup=lineup,
        squad_ids=squad_ids, transfers=transfers, chip=chip,
        bank=round(cfg.budget - value, 1) if mode == 1 else bank,
        squad_value=value, stale=getattr(client, "stale", False),
        trust="Backtest not yet run - treat projections as provisional."
              if mode == 1 else "",
    )
    return rec, xp
```

`run_gameweek.py`:
```python
#!/usr/bin/env python3
"""Single entry point. Refresh data -> model -> optimize -> report.

Runs headless (cron, `claude -p`) with no MCP and no skills.
"""
import argparse
from pathlib import Path

from fpl.config import load_config
from fpl.pipeline import run
from fpl.report.weekly import render

ROOT = Path(__file__).parent


def main() -> int:
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

    rec, xp = run(cfg, mode=args.mode, from_event=args.gw, root=ROOT / "data")
    print(render(rec, xp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass across every task

- [ ] **Step 6: Run against live data end to end**

Run: `python run_gameweek.py --mode 1 --gw 1`
Expected: a full report printed — Starting XI with a valid formation, 4 bench players, captain and vice, "No transfer recommended", chip watch, budget within £100.0m, and a Flags section. Confirm by eye that the squad has ≤3 players per club.

- [ ] **Step 7: Verify the headless path**

Run: `claude -p "run the weekly gameweek update"` from the project directory.
Expected: it invokes `run_gameweek.py` and prints the report. If MCP or skills fail to load in headless mode, that is acceptable and expected — the pure-Python path must still work. Record the outcome in the commit message.

- [ ] **Step 8: Commit**

```bash
git add fpl/pipeline.py run_gameweek.py tests/test_pipeline.py
git commit -m "feat: end-to-end pipeline and headless CLI entry point"
```

---

## Plan Self-Review

**Spec coverage.** Every spec section maps to a task: §5 data layer → Tasks 1–3; §6 xP model → Tasks 4–9 (§6.0 two-scoring-paths → Task 8 docstring + Task 9 DC component; §6.1 shrinkage → Task 7; §6.2 form → Task 7; §6.3 minutes → Task 6; §6.4 strength → Task 4; §6.4b odds → Task 4 `odds_provider`; §6.5 DGW/BGW → Task 5 + Task 9; §6.6 DC verification → Task 9); §7 backtest → Tasks 10–11; §8 optimizer → Tasks 12–13, 15–16; §9 report → Task 17; §10 config → Task 0; §11 headless → Task 18; §12 guardrails → Task 2 (`FORBIDDEN`) and Task 18 (import test); §13 risks → surfaced via `flags` and `trust`.

**Two deliberate deviations, both documented at their task:**
1. Task 12 uses `mean(bench_weight)` in the MILP rather than exact per-slot weights — exact bench ordering needs ordering variables for negligible gain; the per-slot weights are applied in Task 13 for presentation.
2. Task 14 corrects spec §8's FT formula. `min(5, FT + 1 − made)` returns 0 for (FT=2, made=3); the correct answer is 1, because the balance floors at 0 *before* accruing. Implemented as `min(5, max(0, FT − used) + 1)`.

**Known gap, deliberately deferred.** Tasks 10–11 build the backtest machinery and the trust gate, but no task *runs* it against real vaastav data and reports the number — that requires the archive fetch plus a fitting pass over `form_half_life_gw`, `form_max_weight`, and `shrinkage_minutes`. Until then `Recommendation.trust` says "Backtest not yet run — treat projections as provisional" (Task 18). This is a real limitation to close before the 21 Aug deadline, not an oversight.

**Type consistency checked.** `CONTRACT_COLUMNS` is asserted identical in Tasks 9 and 18. `Squad`, `Lineup`, `TransferPlan`, `ChipAdvice`, and `Recommendation` field names are used consistently across Tasks 12–18. `Config` attribute names match Task 0 everywhere they appear.

