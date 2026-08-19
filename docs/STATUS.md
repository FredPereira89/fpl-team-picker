# Status

Running log of what's been decided and done. Read this first when resuming work.

## Current state (as of 2026-08-19)

- Season 2026/27, GW1 deadline **Fri 21 Aug 2026 17:30 UTC**.
- `entry_id: 1461088` (Frederico Pereira, "Haaland&Grosso") set in `config.yaml`.
- User's live GW1 squad checked against the model (Mode 1 rebuild, fresh data pulled 2026-08-19):
  starting XI, bench order, and captain/vice (Haaland C, Thiago VC) matched the model's own
  pick from those 15 players exactly. Full-rebuild alternative only gained +2.84 xP over 5 GWs
  (~0.57/week) — inside the model's own noise band. Verdict: squad holds, no transfer needed.
  Only flagged risk: Yates (bench 4) was 75% chance of playing (unspecified injury) — optional
  swap to D.Essugo for +0.38 xP, not required.
- `data/state.json` (FT balance / chips used) does not exist yet — Mode 2 hasn't recorded a
  gameweek yet, so it will be created on first `--mode 2` run after GW1.
- Backtest (`13afabd`): outfield (DEF/MID/FWD) rank quality beats naive baseline and FPL's own
  xP; goalkeeper rank quality is unproven (Spearman 0.034) — treat GK picks with extra caution.

## Session log

### 2026-08-19
- Refreshed `bootstrap-static` + `fixtures` cache (previous snapshot was from 2026-07-26).
- Ran `run_gameweek.py --mode 1 --gw 1`, validated all 15 model-picked players against live
  `status`/`news` — all available, no flags.
- Compared user's actual GW1 squad (from screenshot) against the model; see Current state above.
- Committed and pushed `.agents/skills/` (mirror of `.claude/skills/` project skills used by
  Claude Code's agent tooling) — `cf6f66f`.
- Created this file.

### Earlier (from git log, not narrated session-by-session)
- `313d898` Fixed mode/trust mislabeling, mode-2 CLI message, unimplemented risk profiles.
- `1c6045d` Added design spec + implementation plan (`docs/superpowers/`) + project skills.
- `0ac6814` Set `entry_id` for Frederico Pereira's team.
- `13afabd` Ran backtest against real data (see Current state above for result).
- `2488bd9` Wired up Mode 2 CLI to fetch live squad, raised bench weighting.
