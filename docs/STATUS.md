# Status

Running log of what's been decided and done. Read this first when resuming work.

## Current state (as of 2026-08-26)

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
- **GW1 is now finished (result recorded 2026-08-25): actual score 54 pts** vs the 60.48
  predicted (−6.48, −10.7%), driven mainly by Haaland's captain blank. See session log below
  for the full breakdown.

### GW2 (deadline Fri 28 Aug 2026 17:30 UTC)

- **Transfer made by the user 2026-08-27: Enzo (CHE) → Tzolis (ARS), £7.0m → £6.5m.**
  Uses the single free transfer, no hit. **0 FT remaining**; any further move costs −4.
- Squad is legal: 2/5/5/3, Arsenal now at the 3-player cap (Raya, Calafiori, Tzolis),
  Man City also at 3. Chelsea drops to 2, freeing a slot there.
- Bank £0.5m. Current market value £99.7m.
- **First price moves of the season, both rises, both ours: Calafiori £5.5 → £5.6 and
  João Pedro £7.5 → £7.6** (as of the 2026-08-27 snapshot). Selling value is unchanged
  either way — a rise has to reach +0.2 before it is worth +0.1 on sale.
- **Recommendation: hold. Start Tzolis. Do not pay −4 to undo the transfer.**
  XI (4-3-3): Raya; Virgil, Guéhi, Calafiori, Tarkowski; Semenyo, Schade, Tzolis;
  Haaland (C), Thiago (VC), João Pedro. Bench: Dubravka, D.Essugo, Kadıoğlu, Hughes.
- The superseded pre-transfer advice was to hold and bank the FT; the only move the model
  liked was Raya → Leno at +1.55 xP/5GW, rejected because GK is the position the backtest
  measured at Spearman 0.034.

### ⚠ The model cannot value Tzolis, and says so misleadingly

With 0 FT left the optimizer recommends **Tzolis → Enzo on a −4 hit**. That recommendation
should not be followed, because the comparison behind it is not real.

Tzolis signed from Club Brugge in July 2026, so he has **no 2025/26 Premier League row**.
`apply_season_baseline` therefore zeroes him, which is right for scoring rates but produces
two separate distortions:

1. **Minutes.** He falls through to the price prior (0.58) and gets flagged "no Premier League
   minutes on record" — flatly wrong: he started GW1, played 75 minutes, scored 6, and is
   25.9% owned. Fixed with an override at 0.85 (`data/overrides.yaml`), which puts him in the XI.
2. **Scoring rates — not fixable by any override.** xG90, xA90, bonus90 and DC90 are all
   shrunk to the *positional mean*. The model has no idea whether he is good; it only knows he
   plays. **Its xP for him is a floor, not an estimate.**

So the headline numbers — Tzolis 14.20 vs Enzo 23.35 over 5 GWs on the raw prior, or 17.0-ish
with the minutes override — compare Enzo's real measured output against a league-average
placeholder. The true gap is smaller and unknown in sign. Enzo also brings a real risk the
model *does* see and price: he lasted 25 minutes in GW1.

Sweep for the record (effective p_start after the 0.5 news blend, then GW2 / 5GW xP):
raw 0.58 → 2.66 / 14.20; override 0.80 → 0.69 eff → 3.08 / 16.42; 0.90 → 0.74 → 3.27 / 17.48;
1.00 → 0.79 → 3.48 / 18.56. He starts the XI at 0.85 and above.

**This is the general new-signing blind spot**, not a Tzolis quirk: any player without a
prior-season PL row is invisible to the rate model and will be systematically undervalued
until enough 2026/27 data accumulates — which needs form blending wired up (see next session).

### Calafiori minutes override — now a real feature, premise re-verified

No longer a scratch-script hack: overrides are declarative in `data/overrides.yaml`, loaded
by `fpl/data/overrides.py`, passed through `run_gameweek.py --overrides`. Each entry requires
a `source` and takes an inclusive `until_gw` so it expires instead of quietly outliving the
news behind it. (This was standing "next session" item 3.)

**`minutes_model` sets an injured player's own p_start to 0 but does NOT redistribute those
minutes to teammates.** It is per-player, not a team-level allocation. This is a structural
blind spot the model cannot see, and it is the whole reason the override mechanism exists.

- **Premise re-verified 2026-08-26: Saliba (back) and J.Timber (groin) are both still
  `status=i`, 0% chance, unknown return.**
- **Validated by events: Calafiori started GW1** (80 mins, clean sheet + assist, 9 pts) while
  Hincapie — the other candidate for the spot — played 9 minutes.
- Model's own prior is still 0.48 (his 2025/26 record: 22 starts / 1697 mins).
- **Break-even re-measured at ~0.57** against the best available replacement. The older 0.696
  figure was specifically against Tarkowski and no longer describes the live comparison.
- At `news.weight: 0.5`, an override of 0.80 yields an **effective 0.64** — clear of the
  break-even, so the conservative default blend is sufficient and `news.weight` was left alone.
  Swept 0.48 → 0.90 to confirm where the recommendation flips.
- Set to expire `until_gw: 6`. **Revisit if Saliba or Timber is passed fit.**

### Other standing notes

- `data/state.json` (FT balance / chips used) is written by each `--mode 2` run. It is
  gitignored, and `reconcile` re-derives the FT count from `entry/{id}/history/` every run
  anyway, so the file only drives the drift warning — deleting it is safe.
- **Form blending is dead code.** `blend_form` / `form_weight` in `model/scoring.py` exist and
  are unit-tested, but nothing in `pipeline.run` calls them. The model is therefore 100%
  last-season baseline and **cannot see 2026/27 form at all**. Biggest remaining gap now that
  the baseline source is fixed.
- **Mode 2 cannot see the squad before a deadline.** `entry/{id}/event/1/picks/` returns
  **HTTP 404** until the deadline passes; `entry/` and `entry/history/` work but carry no picks.
  Verified 2026-08-20. This is why pre-deadline work relies on screenshots. Auth-gated
  `my-team/` is banned by the spec and was not used.
- Backtest (`13afabd`): outfield (DEF/MID/FWD) rank quality beats naive baseline and FPL's own
  xP; goalkeeper rank quality is unproven (Spearman 0.034) — treat GK picks with extra caution.
  Raya at £6.0m rests on that unvalidated component. This is also what drives the GW2 call to
  *not* do Raya → Leno.
- 5-GW clean-sheet projections are directional, not precise, past ~GW2 — they are built from
  last season's team ratings.

## Session log

### 2026-08-26 — three bugs, and GW2

The first live Mode 2 run of the season was badly broken: it wanted 4 transfers on a −8 hit,
recommended **selling Haaland**, and reported a £10.7m bank against a £100.0m squad. Three
distinct bugs, each found by checking output rather than trusting it.

1. **Season-rollover baseline bug (the big one).** `normalize_players` read the model's
   baseline counting stats straight from `bootstrap-static`, which reports **current-season
   cumulative** totals. The FPL API reset those at the 2026/27 rollover — league-wide minutes
   went from **602,348** in the 2026-08-20 snapshot to **19,671** on 2026-08-26. But
   `minutes_model` and `per90_rates` both treat that column as a *full season* (`starts/38`,
   shrinkage k=900). So every established player collapsed to a ~90-minute sample and shrank
   to the positional mean, while anyone who *didn't* feature in GW1 scored **better** by
   falling through to the price prior — hence "sell Haaland, buy Marmoush".
   Fixed by re-sourcing the baseline from `element-summary` `history_past`, which is stable
   all season: new `apply_season_baseline` + `latest_season`, plus
   `FplClient.element_summaries` (30-day TTL — `history_past` for a finished season is
   immutable). Wired unconditionally into `pipeline.run`; pre-season the two sources agree so
   it is a no-op then, avoiding a brittle "has the season started" flag. Cold-cache cost is
   ~10 min for 614 players at the client's deliberate 1 req/s limit; warm runs are seconds,
   and progress is now printed. — `c0bc6bf`
2. **Free-transfer off-by-one.** `state.reconcile` seeded the pre-GW1 balance at 1 FT then
   accrued another after GW1, reporting **2 FTs for GW2**. FPL grants the first FT only
   *after* the GW1 deadline (bootstrap `game_settings` confirms the shape: 1 base +
   `max_extra_free_transfers: 4` = FT_CAP 5), so GW2 has exactly 1. Seeded at 0 instead.
   This one cost real points — it was silently authorising a transfer that actually costs −4.
   Two existing tests encoded the wrong behaviour (`# unused FT rolls 1 -> 2`). — `786b768`
3. **`news` never reached the model from the CLI.** An edit to `run_gameweek.py` silently
   failed to land, so the override loaded, printed, and did nothing — the run still
   succeeded, which is exactly why it was invisible. Caught by noticing the recommendation
   was byte-identical with and without the override. Added
   `test_news_override_reaches_the_minutes_model` at the pipeline layer; `minutes_model`'s own
   override handling was already unit-tested and the untested gap was the wire above it.

Also: made the minutes override a first-class feature (see section above), re-verified the
Calafiori premise against live data, and re-measured its break-even at ~0.57.

Suite 189 → 200 passing; each of the four commits verified green independently.

**GW2 call: hold and bank the FT.** See "Current state" above for the reasoning.


### 2026-08-25 — GW1 retrospective

GW1 fixtures all finished (event `data_checked` still false — minor bonus-point revisions
possible but unlikely to move much). Pulled `entry/1461088/history/` and
`entry/1461088/event/1/picks/` plus `bootstrap-static` for the actual per-player lines.

- **Final score: 54 points** (0 transfers, 0 hits, no chip, bench scored 0 so no autosub
  regret). GW1 average was 50, so +4 above average. Highest score in the game was 131.
- **Overall rank 3,208,418 / 8,903,411 (top 36%)** after GW1.
- Predicted (model, pre-deadline) **60.48** vs actual **54.0** → **−6.48 (−10.7%)**, well
  within normal week-to-week model variance and almost entirely explained by the captain pick.

Per-player actuals (XI only, mult × pts):
| Player | Min | Return | Pts |
|---|---|---|---|
| João Pedro (FWD) | 90 | 1 goal, 1 assist | **11** |
| Guéhi (DEF) | 90 | 1 goal (no CS, conceded) | **10** |
| Calafiori (DEF) | 80 | 1 assist, clean sheet | **9** |
| Raya (GKP) | 90 | clean sheet, 1 save | 6 |
| Tarkowski (DEF) | 90 | clean sheet | 6 |
| Schade (MID) | 90 | clean sheet (attacker, no bonus) | 3 |
| Haaland (FWD, **C**) | 90 | blank | **2 → 4** (captained) |
| Virgil (DEF) | 90 | blank, yellow card | 2 |
| Semenyo (MID) | 90 | blank | 2 |
| Enzo (MID) | 25 | subbed off early, blank | 1 |
| Thiago (FWD, VC) | 82 | **missed a penalty** (2 appearance − 2 miss) | 0 |

- **Calafiori minutes override validated.** The 0.80 `p_start` override (vs. the model's own
  0.48 prior) called it right: he started, played 80 minutes, got a clean sheet + assist for 9
  points. Good evidence the Saliba/Timber-injury reasoning was sound, at least for week 1.
- **Captaincy was the biggest drag.** Haaland (most-captained player in the game, so no rank
  damage relative to the template) blanked at 2 base points — a top scorer with an off week
  everyone shares in. Thiago (VC) would also have blanked as captain (missed penalty voided his
  appearance points).
- Bench (Dubravka, Kadıoğlu, Hughes, D.Essugo) all played 0 minutes — nothing missed there.
- Squad value unchanged at exactly £100.0m, £0.0m bank — no price rises/falls banked yet.

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

1. **Tzolis is now owned** (bought GW2, replacing Enzo). Outstanding question is not whether
   to buy but whether the model can ever score him properly — see the blind-spot section above.
   Re-check after a few GWs of 2026/27 data accumulate. His minutes override expires at GW6.
2. Re-check Arsenal team news before the GW2 deadline (Saliba/Timber → Calafiori premise,
   currently overridden to 0.80 and expiring at GW6).
3. **Wire up form blending.** `blend_form` / `form_weight` are tested but uncalled, so the
   model still cannot see any 2026/27 form. This is the biggest remaining modelling gap —
   and it grows every gameweek.
4. Consider whether the GK component is worth using at all given Spearman 0.034, or whether
   GK picks should just defer to FPL's own xP / team news.
