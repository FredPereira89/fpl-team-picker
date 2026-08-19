---
name: weekly-report
description: Use when presenting the final gameweek recommendation to the user - starting XI, bench order, captain, transfers, and chip advice. Trigger on "give me this week's team", "show my gameweek report", "what's my lineup", or right after the squad-optimizer skill produces a result.
---

# Weekly Report

Takes the output of `squad-optimizer` (and the underlying xP figures) and turns it into a short, readable report. Don't run models here — just present.

## Format

```
## Gameweek {N} — {deadline date/time}

### Starting XI ({formation})
GK: ...
DEF: ...
MID: ...
FWD: ...

### Bench (in order)
1. ...
2. ...
3. ...
4. ...

### Captain: {player} (C)  |  Vice: {player} (VC)
One line on why this player over the alternatives.

### Transfers this week
{OUT} → {IN}: net xP gain of {X} over next {N} GWs (after any -4 hit). / "No transfer recommended — squad already optimal."

### Chip watch
{Recommendation or "No chip recommended this week"} — one line of reasoning.

### Budget
Bank: £{X}m | Squad value: £{Y}m

### Flags
Any injury/rotation/news uncertainty on players in the squad, with source.
```

## Rules
- One sentence of "why" per major decision (captain, each transfer) — never just numbers with no explanation
- Always surface uncertainty explicitly rather than presenting a guess as settled fact
- Never phrase anything as already done — this is a recommendation for the user to apply manually in the FPL app, so use "recommend", "suggest", not "transferred" or "captained"
- Keep it scannable — this is meant to be read in under a minute before a deadline
