# Status

Running log of what's been decided and done. Read this first when resuming work.

## Current state (as of 2026-08-20)

- Season 2026/27, GW1 deadline **Fri 21 Aug 2026 17:30 UTC**.
- `entry_id: 1461088` (Frederico Pereira, "Haaland&Grosso") set in `config.yaml`.
- **GW1 squad is FINAL** (locked 2026-08-20, £100.0m exactly, £0.0m bank, 0 flagged players):
  - GKP: Raya (ARS), Dubravka (TOT)
  - DEF: Calafiori (ARS), Guéhi (MCI), Virgil (LIV), Tarkowski (EVE), F.Kadıoğlu (BHA)
  - MID: Semenyo (MCI), Schade (BRE), Enzo (CHE), Hughes (CRY), D.Essugo (CHE)
  - FWD: Haaland (MCI), Thiago (BRE), João Pedro (CHE)
  - XI 4-3-3: Raya; Calafiori, Guéhi, Virgil, Tarkowski; Semenyo, Schade, Enzo;
    Haaland (C), Thiago (VC), João Pedro. Bench: Dubravka, Kadıoğlu, Hughes, D.Essugo.
  - Man City and Chelsea both at the 3-player cap — no room to add from either.
- Scores 248.88 on the model's 5-GW objective vs a 249.04 unconstrained optimum
  (**gap 0.16**) once the Calafiori minutes override below is applied. GW1 xP 60.48.

### ⚠ Open assumption: Calafiori minutes override

The model's own prior gives Calafiori `p_start = 0.48` (from 22 starts / 1697 mins in 2024/25).
We are deliberately overriding this to **0.80** because Saliba (back) and J.Timber (groin) are
both `status=i`, 0% chance, unknown return — removing 58 starts of competition.

**`minutes_model` sets an injured player's own p_start to 0 but does NOT redistribute those
minutes to teammates.** It is per-player, not a team-level allocation. This is a structural
blind spot, not a bug we introduced — the model cannot see the Arsenal injury situation at all.

- **Break-even is `p_start = 0.696`.** Above it Calafiori beats Tarkowski; below it he does not.
- At 0.48 he is xp5 14.34 (model wants him sold). At 0.80 he is 21.50 — beats Tarkowski by
  +2.92 and even beats Gabriel (20.64) at £2.5m less.
- **Revisit when Saliba or Timber is passed fit.** If either returns the premise weakens.
  Also note Hincapie (£5.5m, 20 starts, xp5 12.03) is genuine competition for the spot, and
  part of Calafiori's low start count is his own injury history, which their absence doesn't fix.

### Other standing notes

- `data/state.json` (FT balance / chips used) does not exist yet — Mode 2 hasn't recorded a
  gameweek. It gets created on the first `--mode 2` run after GW1 completes.
- **Mode 2 cannot see the squad before a deadline.** `entry/{id}/event/1/picks/` returns
  **HTTP 404** until the deadline passes; `entry/` and `entry/history/` work but carry no picks.
  Verified 2026-08-20. This is why pre-deadline work relies on screenshots. Auth-gated
  `my-team/` is banned by the spec and was not used.
- Backtest (`13afabd`): outfield (DEF/MID/FWD) rank quality beats naive baseline and FPL's own
  xP; goalkeeper rank quality is unproven (Spearman 0.034) — treat GK picks with extra caution.
  Raya at £6.0m rests on that unvalidated component.
- 5-GW clean-sheet projections are directional, not precise, past ~GW2 — they are built from
  last season's team ratings.

## Session log

### 2026-08-20
- Refreshed data again (deadline ~29h out). Re-scored after user made two transfers.
- **Gabriel → Calafiori and Calvert-Lewin → João Pedro.** João Pedro was an upgrade (+3.50);
  Calafiori looked like −6.30 on the model's prior, until the Arsenal injury news reframed it.
- Decomposed the Calafiori/Tarkowski gap into appear/goals/CS/DC components. Established that
  the user's "Arsenal keep more clean sheets" argument was **correct** — at equal minutes
  Calafiori's CS points beat Tarkowski's 8.84 to 7.28 — and that the disagreement was entirely
  about minutes, not clean sheets. Derived the 0.696 break-even.
- Verified Saliba and Timber both out via live API; adopted the 0.80 override.
- Evaluated and rejected two alternatives to Tarkowski:
  - **Lacroix** (now at **Chelsea**, not Palace): best DC/90 in the group at 10.79 and best
    GW1 xp1 (4.08), but worst 5-GW fixtures — Σp_cs 1.43 vs Everton's 2.19, includes @ARS at
    p_cs 0.06 — and would have taken Chelsea to the 3-cap. xp5 15.66.
  - **Muñoz** (Crystal Palace, id 201 — not the Liverpool "Munoz" id 377 with 0 PL minutes):
    best attacking output (8 G+A) but DC/90 7.35 misses the threshold, p_start 0.64. xp5 13.06.
  - Both score *below* the N.Williams (16.57) they'd have replaced. Tarkowski (18.59) was the
    only gaining move.
- Applied final two transfers: **N.Williams → Tarkowski** (+2.02, used the £1.0m bank) and
  **Yates → D.Essugo** (cleared the last injury flag). Squad locked.
- Bench slots 3/4: model marginally prefers D.Essugo (1.52) over Hughes (1.40), but Essugo's
  estimate comes from 26 PL minutes and is price-prior driven. Deliberately left as-is.

### 2026-08-19
- Refreshed `bootstrap-static` + `fixtures` cache (previous snapshot was from 2026-07-26).
- Ran `run_gameweek.py --mode 1 --gw 1`, validated all 15 model-picked players against live
  `status`/`news` — all available, no flags.
- Compared user's then-squad against the model: XI, bench order and captaincy already matched;
  a full 7-player rebuild was worth only +2.84 xP over 5 GWs (~0.57/week). Verdict: hold.
- Committed and pushed `.agents/skills/` — `cf6f66f`. Created this file — `d883ede`.

### Earlier (from git log, not narrated session-by-session)
- `313d898` Fixed mode/trust mislabeling, mode-2 CLI message, unimplemented risk profiles.
- `1c6045d` Added design spec + implementation plan (`docs/superpowers/`) + project skills.
- `0ac6814` Set `entry_id` for Frederico Pereira's team.
- `13afabd` Ran backtest against real data (see standing notes above for result).
- `2488bd9` Wired up Mode 2 CLI to fetch live squad, raised bench weighting.

## Next session

1. **Note for future GWs: Tzolis.** Christos Tzolis signed for Arsenal (£34m from Club Brugge,
   replacing Trossard) in July 2026 — not Nottingham Forest. FPL price £6.5m (MID). Ownership
   climbed fast pre-GW1 (~2% early Aug → 20-24%+ by deadline day per Crowd FPL/Beat FPL), on the
   back of a strong pre-season (1G 2A). GW1 squad was already locked when this came up (deadline
   day, API unreachable from this session), so no action was taken — flagged here in case he's
   still a live differential/consideration for a future transfer. Would push Arsenal to the
   3-player cap alongside Raya + Calafiori. Re-run the actual model once data access works before
   acting on this — don't chase ownership on its own.
2. Re-check Arsenal team news before the deadline (Saliba/Timber status → Calafiori premise).
2. After GW1 completes, `entry/1461088/event/1/picks/` goes live — run
   `python run_gameweek.py --mode 2 --gw 2` and it will resolve the real squad automatically.
3. Consider making the Calafiori-style minutes override a first-class feature rather than a
   scratch-script hack: `minutes_model` already accepts a `news` dict with `p_start_override`,
   and `config.yaml` has `news.weight: 0.5` — but nothing populates it. A small
   `data/overrides.yaml` read into `pipeline.run(news=...)` would make team-news corrections
   reproducible instead of ad hoc.
