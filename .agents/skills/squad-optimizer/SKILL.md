---
name: squad-optimizer
description: Use when selecting or re-optimizing an FPL squad - building a wildcard/free-hit squad from scratch, deciding this gameweek's transfer(s), picking captain and vice-captain, or evaluating chip timing. Trigger on "optimize my squad", "who should I transfer", "pick my team", "should I use my wildcard/bench boost/triple captain".
---

# Squad Optimizer

Requires the xP (expected points) model output for the relevant gameweek range — run/refresh that first if it's missing or stale. Requires current data via the `fpl-data-fetch` skill.

## Library
Use `pulp` (MILP) for all squad-selection problems. Install with `pip install pulp` if not already in the project.

## Mode 1 — Full squad build (wildcard / free hit / initial team)
Maximize total expected points over the chosen horizon (1 GW for Free Hit, 5 GW for Wildcard) subject to:
- Budget ≤ £100.0m (or the user's actual bank + squad value if rebuilding mid-season)
- Exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
- Max 3 players from any single real club
- From the 15, select a valid starting XI: 1 GK + (3–5) DEF + (2–5) MID + (1–3) FWD = 11, maximizing xP among feasible formations
- Captain = highest-xP starter, contributes 2× points; vice-captain = second-highest as fallback

## Mode 2 — Weekly transfer optimization
Given the user's current 15-man squad (from `entry/{id}/event/{gw}/picks/`) and free transfers available:
- Search over 0, 1, 2, ... transfers (cap search at a sensible number, e.g. 3, for runtime)
- Each transfer beyond the free allowance costs -4 points
- Maximize: (net xP gain over the horizon) − (4 × paid transfers)
- Always also report the "0 transfers" baseline so the user can see what they'd gain by moving at all
- Respect the same budget/formation/club-limit constraints as Mode 1, applied to the post-transfer squad

## Chip advisor (heuristic, not optimized)
Flag candidate windows, don't auto-recommend without explaining the tradeoff:
- **Wildcard**: fixture swing (a run of good fixtures starting) or squad has multiple injuries/rotation risks needing simultaneous fixing
- **Bench Boost**: use only when all 15 players have good fixtures and healthy bench — check bench players' xP is not near-zero
- **Triple Captain**: single standout player with a double gameweek or a very favorable single fixture
- **Free Hit**: a blank gameweek (many teams have no fixture) or a one-off double gameweek that doesn't fit a longer-term wildcard hold

## Output
Return: the recommended 15 (with position, price), starting XI, bench order, captain/vice, and for Mode 2 the specific transfer(s) with net xP gain after any hit. Hand this off to the `weekly-report` skill for final formatting — don't format the final user-facing report yourself.
