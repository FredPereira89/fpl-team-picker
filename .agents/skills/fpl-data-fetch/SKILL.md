---
name: fpl-data-fetch
description: Use when fetching, refreshing, or caching FPL data - bootstrap-static, fixtures, element-summary, or the user's own entry/picks. Trigger on requests like "refresh the FPL data", "pull the latest gameweek", "update prices and ownership", or before any modeling/optimization step that needs current data.
---

# FPL Data Fetch

## Endpoints
Base URL: `https://fantasy.premierleague.com/api/`

| Endpoint | Use |
|---|---|
| `bootstrap-static/` | players, teams, gameweek/event metadata, prices, ownership, injury flags |
| `fixtures/` and `fixtures/?event={gw}` | fixture list + FDR |
| `element-summary/{player_id}/` | per-player gameweek history + upcoming fixtures |
| `entry/{team_id}/` | my team's basic info |
| `entry/{team_id}/history/` | season + past-season summary |
| `entry/{team_id}/event/{gw}/picks/` | my squad for a given gameweek |

None of these require authentication. Never touch an endpoint that needs a login/session cookie (e.g. `my-team/`), and never write/POST anything.

## Caching rules
1. Before fetching, check `data/cache/` for a file for that endpoint with a timestamp less than 6 hours old (or less than 1 hour old on matchdays). If fresh, reuse it — don't re-fetch.
2. Store raw JSON as `data/cache/{endpoint_slug}_{ISO8601_timestamp}.json`. Keep the last 3 snapshots per endpoint, delete older ones.
3. After fetching, normalize into pandas DataFrames and persist to `data/processed/*.parquet` (or SQLite tables if a database is already set up in this project) — one table per entity: `players`, `teams`, `fixtures`, `player_gw_history`, `my_squad`.
4. Rate-limit: no more than ~1 request/second, and batch `element-summary` calls sequentially with a small delay — this is an unofficial API, be a polite citizen.

## Normalization notes
- `bootstrap-static` returns team and player IDs that must be joined against `teams` and `element_types` (positions) to be human-readable — always resolve IDs to names before handing data to another skill or reporting to the user.
- `now_cost` is in tenths of a million (e.g. `55` = £5.5m) — convert on ingest.
- Flag any player with `status` != `a` (available) — `d`=doubtful, `i`=injured, `s`=suspended, `u`=unavailable — and carry the `news` field through, it often has the return-date estimate.

## Supplementary (non-API) signal
For anything the FPL API doesn't cover — underlying xG/xA, press-conference team news, price-change predictions — use web search/fetch. Always:
- Cite the source when you report a stat back to the user
- Never fabricate or estimate a number if you can't find it — say it's missing instead
- Treat this as a secondary signal layered on top of the hard FPL data, not a replacement for it

## Error handling
If a request fails or the API is down, fall back to the most recent cached snapshot and tell the user the data may be stale, rather than blocking the whole pipeline.
