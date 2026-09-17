"""Prediction ledger: persist each gameweek's forecast, then score it.

Tier 1 and Tier 2 backtests measure proxies of the model. This measures the
model itself: exactly the frame the optimizer consumed, joined against what
actually happened. Nothing else in the codebase records a forecast, so until
now every week's projection was discarded before it could be checked -- the
2026-08-27 audit was only possible because an old bootstrap snapshot happened
to still be sitting in the cache.

Positional BIAS is reported alongside rank quality on purpose. Rank tells you
whether the ordering is right; bias tells you whether the optimizer is being
handed inflated numbers to spend its budget against, which is a different
failure and the one that was live for goalkeepers.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .gw_level import evaluate_predictions
from . import manifest

LEDGER_DIR = "predictions"
VERSION_DIR = "versions"
# Sub-second, because two writes inside the same second are not the same
# forecast -- a planning run immediately followed by a confirmation run would
# otherwise share a version id and silently overwrite each other's record.
TS_FMT = "%Y%m%dT%H%M%S%fZ"
# Derived from the commit that produced the forecast (see manifest.model_version).
# It used to be a date maintained by hand, which had fallen behind several core
# commits -- so scored gameweeks were attributed to a model that was not the one
# that produced them.
MODEL_VERSION = manifest.model_version()


def config_fingerprint(cfg) -> str:
    """Short, stable hash of the settings a forecast was produced under.

    Two runs of the same model with different horizons, decay or shrinkage are
    different forecasts, and a ledger that cannot tell them apart cannot
    attribute a change in score to anything.
    """
    if cfg is None:
        return ""
    data = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    blob = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def save_predictions(xp: pd.DataFrame, gw: int, root: Path, cfg=None,
                     sources: dict | None = None, created_at=None,
                     origin: str = manifest.LIVE,
                     deadline: str | None = None) -> Path:
    """Write one gameweek's xP frame, keep an immutable copy, and record it.

    `gw{n}.parquet` remains a convenience pointer at the most recent write. It
    is NOT authoritative any more: `load_predictions` resolves through the
    append-only manifest instead, because the pointer was being overwritten by
    post-deadline re-runs and by replays, either of which destroyed the record
    of what the optimizer actually acted on.

    `origin` separates a real pre-deadline forecast from one reconstructed
    afterwards. A replay still gets written and versioned -- it is useful for
    research -- but it can never be served as the gameweek's forecast.
    """
    out = Path(root) / LEDGER_DIR
    out.mkdir(parents=True, exist_ok=True)
    when = created_at or datetime.now(timezone.utc)
    version = when.strftime(TS_FMT)

    frame = xp.copy()
    frame.insert(0, "gw", int(gw))
    frame["created_at"] = when.isoformat()
    frame["model_version"] = MODEL_VERSION
    frame["config_hash"] = config_fingerprint(cfg)
    frame["sources"] = json.dumps(sources or {}, sort_keys=True)
    frame["origin"] = str(origin)

    versions = out / VERSION_DIR
    versions.mkdir(parents=True, exist_ok=True)
    version_path = versions / f"gw{int(gw)}_{version}.parquet"
    frame.to_parquet(version_path, index=False)

    # The pointer only follows a live write. A replay that moved it would make
    # every reader that predates the manifest read the replay instead.
    if str(origin) == manifest.LIVE:
        frame.to_parquet(out / f"gw{int(gw)}.parquet", index=False)

    manifest.record_version(root, gw=int(gw), version=version, created_at=when,
                            origin=str(origin), model_version=MODEL_VERSION,
                            config_hash=config_fingerprint(cfg),
                            deadline=deadline)
    return version_path


def forecast_versions(gw: int, root: Path) -> list[Path]:
    """Every recorded version of one gameweek's forecast, oldest first."""
    d = Path(root) / LEDGER_DIR / VERSION_DIR
    if not d.exists():
        return []
    return sorted(d.glob(f"gw{int(gw)}_*.parquet"))


def gameweek_is_final(fixtures: list[dict], gw: int) -> bool:
    """True once every fixture in `gw` has been checked by FPL.

    `finished_provisional` flips at the final whistle; `finished` only after
    the bonus and stat review, which can move points hours later. Scoring a
    forecast against provisional returns measures the model against numbers
    that are still moving, so the verdict has to say which it is.
    """
    rows = [f for f in (fixtures or []) if int(f.get("event") or 0) == int(gw)]
    return bool(rows) and all(bool(f.get("finished")) for f in rows)


def load_predictions(gw: int, root: Path, deadline: str | None = None) -> pd.DataFrame:
    """The forecast that should be scored for `gw`.

    Resolved through the manifest -- the acted-on version, else the newest live
    version before the deadline -- and never a replay. Falls back to the
    `gw{n}.parquet` pointer only for ledgers written before the manifest
    existed, which have no version history to choose from.
    """
    chosen = manifest.select_version(root, int(gw), deadline=deadline)
    if chosen is not None:
        path = (Path(root) / LEDGER_DIR / VERSION_DIR
                / f"gw{int(gw)}_{chosen['version']}.parquet")
        if path.exists():
            return pd.read_parquet(path)

    pointer = Path(root) / LEDGER_DIR / f"gw{int(gw)}.parquet"
    if not pointer.exists():
        raise FileNotFoundError(
            f"no live forecast on file for GW{gw}. A replay-only gameweek is "
            f"deliberately not served: it was built from data the live model "
            f"never had, so scoring or calibrating on it measures the wrong "
            f"model."
        )
    if manifest.entries(root, int(gw)):
        # The manifest knows this gameweek and chose nothing, so every recorded
        # version is a replay. The stale pointer must not stand in for one.
        raise FileNotFoundError(
            f"GW{gw} has only replayed forecasts on file, which are never "
            f"served as a live gameweek's record."
        )
    return pd.read_parquet(pointer)


def available_gameweeks(root: Path) -> list[int]:
    d = Path(root) / LEDGER_DIR
    if not d.exists():
        return []
    return sorted(int(p.stem[2:]) for p in d.glob("gw*.parquet"))


def actuals_from_summaries(summaries: dict[int, dict], gw: int) -> pd.DataFrame:
    """Actual points and minutes for one gameweek, from element-summary history.

    A double gameweek gives a player two rows in the same round, and FPL scores
    both, so rows are summed rather than deduplicated.
    """
    rows = []
    for pid, summary in (summaries or {}).items():
        for h in summary.get("history", []):
            if int(h.get("round", -1)) == int(gw):
                rows.append({
                    "player_id": int(pid),
                    "actual": float(h.get("total_points", 0)),
                    "minutes": float(h.get("minutes", 0)),
                })
    if not rows:
        return pd.DataFrame(columns=["player_id", "actual", "minutes"])
    return pd.DataFrame(rows).groupby("player_id", as_index=False).sum()


CANDIDATE_POOL = 60
# Rankers the model is measured against. `price` is the market's own consensus
# and is the baseline that matters: a model that cannot beat "buy the expensive
# ones" is not earning its complexity. `p_start` isolates how much of the
# ordering is just telling starters from reserves.
BASELINE_RANKERS = ["price", "p_start"]
N_BOOT = 2000
# Gameweeks of scored history before a verdict means much. Below this the
# interval on a single-gameweek rank correlation is wider than the entire
# range of plausible skill.
MIN_GAMEWEEKS_FOR_A_VERDICT = 5


def _rho(frame: pd.DataFrame, pred_col: str) -> float:
    if len(frame) < 3 or frame[pred_col].nunique() < 2 or frame["actual"].nunique() < 2:
        return 0.0
    r = spearmanr(frame[pred_col].values, frame["actual"].values).statistic
    return 0.0 if np.isnan(r) else float(r)


def spearman_on(frame: pd.DataFrame, pred_col: str) -> float:
    """Rank correlation of one ranker against actuals on a GIVEN pool."""
    return _rho(frame, pred_col)


def common_pool(df: pd.DataFrame, rankers: list[str], top_n: int = CANDIDATE_POOL) -> pd.DataFrame:
    """The union of every ranker's top `top_n`.

    Scoring a ranker on the pool IT selected is range restriction: the
    predictor's variance collapses inside its own top slice while the outcome's
    widens, so the correlation is attenuated by construction. On GW2 the model
    scored +0.114 on its own top 60 and +0.431 on a pool chosen this way -- the
    same forecast, the same actuals.

    Taking the union rather than one ranker's slice also means no candidate is
    scored on a pool it chose, so the comparison is symmetric.
    """
    present = [c for c in rankers if c in df.columns]
    if not present:
        return df
    keep = pd.concat([df.nlargest(top_n, c) for c in present])
    return keep.drop_duplicates(subset="player_id").reset_index(drop=True)


def compare_rankers(df: pd.DataFrame, rankers: list[str],
                    top_n: int = CANDIDATE_POOL) -> dict:
    """{ranker: {rho, lo, hi, n}} with every candidate scored on ONE pool."""
    pool = common_pool(df, rankers, top_n)
    out = {}
    for c in rankers:
        if c not in pool.columns:
            continue
        rho, lo, hi = spearman_ci(pool, c)
        out[c] = {"rho": rho, "lo": lo, "hi": hi, "n": int(len(pool))}
    return out


def spearman_ci(frame: pd.DataFrame, pred_col: str, n_boot: int = N_BOOT,
                seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(rho, lo, hi) -- a bootstrap percentile interval on the rank correlation.

    A bare rho invites over-reading a single gameweek. On GW2 the headline
    top-60 figure carried a 95% interval of [-0.15, +0.37]: it could not
    distinguish no skill from good skill, and was nonetheless printed as the
    verdict on the model.
    """
    f = frame[[pred_col, "actual"]].dropna()
    point = _rho(frame, pred_col)
    if len(f) < 5:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = f.values
    stats = []
    for _ in range(int(n_boot)):
        sample = values[rng.integers(0, len(values), len(values))]
        if len(np.unique(sample[:, 0])) < 2 or len(np.unique(sample[:, 1])) < 2:
            continue
        r = spearmanr(sample[:, 0], sample[:, 1]).statistic
        if not np.isnan(r):
            stats.append(r)
    if not stats:
        return point, float("nan"), float("nan")
    return (point, float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def score_gameweek(pred: pd.DataFrame, actuals: pd.DataFrame,
                   pred_col: str = "xp_next1", top_n: int = CANDIDATE_POOL) -> dict:
    """Score one gameweek's forecast against what happened."""
    df = pred.merge(actuals, on="player_id", how="inner")
    if len(df) == 0:
        raise ValueError(
            "cannot score this gameweek: the prediction frame and the actuals "
            "have no players in common (wrong gameweek, or the gameweek has "
            "not been played yet)"
        )

    err = df[pred_col].astype(float) - df["actual"].astype(float)
    base = evaluate_predictions(df[pred_col], df["actual"], df["position"])

    played = df[df["minutes"] > 0]
    absent = df[df["minutes"] <= 0]

    return {
        **base,
        "bias": float(err.mean()),
        "bias_by_position": {
            pos: float((g[pred_col].astype(float) - g["actual"].astype(float)).mean())
            for pos, g in df.groupby("position")
        },
        # Rank quality among players who actually appeared isolates the scoring
        # model from the minutes model, which is where the two-layer diagnosis
        # in the 2026-08-27 audit came from.
        "spearman_played": _rho(played, pred_col),
        "n_played": int(len(played)),
        # Kept for continuity with earlier scores, but do NOT read it as skill:
        # selecting the top N by the model and then correlating the model
        # inside that slice attenuates the correlation by construction. Use
        # `ranker_comparison`, which scores every candidate on one shared pool.
        "spearman_top_n": _rho(df.nlargest(top_n, pred_col), pred_col),
        "n_top": int(min(top_n, len(df))),
        # The honest question: on one pool nobody selected for themselves, does
        # the model order players better than the market does?
        "ranker_comparison": compare_rankers(df, [pred_col] + BASELINE_RANKERS, top_n),
        "spearman_overall_ci": spearman_ci(df, pred_col)[1:],
        # Splitting bias this way names the layer at fault. A position looks
        # over-predicted when players who never featured were handed points,
        # which is the minutes model; the scoring model is only implicated by
        # bias among those who did play.
        "bias_played": float((played[pred_col].astype(float)
                              - played["actual"].astype(float)).mean())
        if len(played) else 0.0,
        "bias_absent": float(absent[pred_col].astype(float).mean())
        if len(absent) else 0.0,
        "n_absent": int(len(absent)),
    }


def scored_summary(scored: dict, gw: int, provisional: bool = False) -> str:
    """One-screen verdict, written to be read by someone deciding whether to
    trust this week's recommendation."""
    by_pos = scored["spearman_by_position"]
    bias = scored["bias_by_position"]
    lines = []
    if provisional:
        # Bonus points and stat corrections still move after the final whistle.
        lines.append(
            f"PROVISIONAL: GW{gw} is not fully checked by FPL yet, so these "
            f"numbers will still move. Re-score once it is final."
        )
    lo, hi = scored.get("spearman_overall_ci", (float("nan"), float("nan")))
    lines += [
        f"GW{gw} forecast scored against {scored['n']} players "
        f"({scored['n_played']} of them appeared):",
        f"  rank quality (Spearman)  {scored['spearman_overall']:+.3f} overall "
        f"(95% CI {lo:+.3f}..{hi:+.3f}), "
        f"{scored['spearman_played']:+.3f} among players who appeared",
        f"  ...in its own top {scored['n_top']:<3}      "
        f"{scored['spearman_top_n']:+.3f}  <- RANGE-RESTRICTED, not a skill "
        f"estimate; see the comparison below",]
    comparison = scored.get("ranker_comparison") or {}
    if comparison:
        n_pool = next(iter(comparison.values()))["n"]
        lines.append(f"  vs baselines on one shared pool of {n_pool} players:")
        for name, s in sorted(comparison.items(), key=lambda kv: -kv[1]["rho"]):
            label = "THIS MODEL" if name not in BASELINE_RANKERS else name
            lines.append(f"    {label:<12} {s['rho']:+.3f}  "
                         f"(95% CI {s['lo']:+.3f}..{s['hi']:+.3f})")
    lines += [
        f"  error                    MAE {scored['mae']:.2f}, RMSE {scored['rmse']:.2f}, "
        f"bias {scored['bias']:+.2f} pts/player",
        f"  bias split               {scored['bias_played']:+.2f} among players who "
        f"appeared, {scored['bias_absent']:+.2f} handed to the "
        f"{scored['n_absent']} who did not play",
        f"  top-20 overlap           {scored['top20_overlap']:.0%}",
        "  by position:",
    ]
    for pos in sorted(set(by_pos) | set(bias)):
        lines.append(
            f"    {pos:<4} rank {by_pos.get(pos, float('nan')):+.3f}   "
            f"bias {bias.get(pos, float('nan')):+.2f}"
        )
    over = [p for p, b in bias.items() if b > 0.4]
    if over:
        # Which layer is at fault depends on where the bias sits. Points given
        # to players who never featured are a minutes/news failure; the scoring
        # model is only implicated by bias among those who actually appeared.
        culprit = ("the minutes model — most of it is points handed to players who "
                   "did not play, not inflated scoring"
                   if scored["bias_absent"] > abs(scored["bias_played"])
                   else "the scoring model — players who appeared were over-rated")
        lines.append(
            "  WARNING: over-predicting " + ", ".join(sorted(over)) +
            f" by more than 0.4 pts/player. Look at {culprit}."
        )
    # The intervals above are wide for a reason. Rank correlation on one
    # gameweek of one pool is a noisy statistic, and acting on a single week's
    # movement is how a working model gets tuned into a worse one.
    lines.append(
        f"  Read this as one gameweek, not a verdict: the intervals overlap "
        f"heavily and {MIN_GAMEWEEKS_FOR_A_VERDICT}+ scored gameweeks are "
        f"needed before a change in these numbers means anything."
    )
    return "\n".join(lines)


SCORE_FILE = "last_score.txt"


def save_scored_summary(text: str, root: Path) -> Path:
    """Persist the most recent scoring verdict so the next run can quote a real
    measurement instead of a hard-coded claim about a proxy backtest."""
    out = Path(root) / LEDGER_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / SCORE_FILE
    path.write_text(text, encoding="utf-8")
    return path


def load_scored_summary(root: Path) -> str | None:
    path = Path(root) / LEDGER_DIR / SCORE_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None
