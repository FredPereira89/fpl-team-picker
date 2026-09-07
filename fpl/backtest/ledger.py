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

LEDGER_DIR = "predictions"
VERSION_DIR = "versions"
TS_FMT = "%Y%m%dT%H%M%SZ"
# Bumped by hand when the forecast changes in a way that makes older scores
# incomparable. A scored gameweek says which model produced it or it measures
# nothing in particular.
MODEL_VERSION = "2026-09-07"


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
                     sources: dict | None = None, created_at=None) -> Path:
    """Write one gameweek's xP frame, and keep an immutable copy of it.

    `gw{n}.parquet` is the current forecast: a re-run before the deadline
    supersedes the earlier one, which is what the scorer and the report want.
    But overwriting also destroyed every earlier version, so "the forecast the
    optimizer acted on" was unrecoverable the moment anything was re-run --
    including the pre-deadline forecast that a mid-week team-news update
    replaced. Each write therefore also lands under `versions/` under its own
    timestamp, and carries when it was made, which model and config made it,
    and which data snapshots it read.
    """
    out = Path(root) / LEDGER_DIR
    out.mkdir(parents=True, exist_ok=True)
    when = created_at or datetime.now(timezone.utc)

    frame = xp.copy()
    frame.insert(0, "gw", int(gw))
    frame["created_at"] = when.isoformat()
    frame["model_version"] = MODEL_VERSION
    frame["config_hash"] = config_fingerprint(cfg)
    frame["sources"] = json.dumps(sources or {}, sort_keys=True)

    path = out / f"gw{int(gw)}.parquet"
    frame.to_parquet(path, index=False)

    versions = out / VERSION_DIR
    versions.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(versions / f"gw{int(gw)}_{when.strftime(TS_FMT)}.parquet",
                     index=False)
    return path


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


def load_predictions(gw: int, root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / LEDGER_DIR / f"gw{int(gw)}.parquet")


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


def _rho(frame: pd.DataFrame, pred_col: str) -> float:
    if len(frame) < 3 or frame[pred_col].nunique() < 2 or frame["actual"].nunique() < 2:
        return 0.0
    r = spearmanr(frame[pred_col].values, frame["actual"].values).statistic
    return 0.0 if np.isnan(r) else float(r)


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
        # Pool-wide rank quality mostly measures telling starters from reserves,
        # which is not a decision anyone needs help with. Squads are picked from
        # the top of the model's own ordering, so skill has to be reported there
        # too -- GW2 came out +0.595 overall and +0.114 inside its own top 60.
        "spearman_top_n": _rho(df.nlargest(top_n, pred_col), pred_col),
        "n_top": int(min(top_n, len(df))),
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
    lines += [
        f"GW{gw} forecast scored against {scored['n']} players "
        f"({scored['n_played']} of them appeared):",
        f"  rank quality (Spearman)  {scored['spearman_overall']:+.3f} overall, "
        f"{scored['spearman_played']:+.3f} among players who appeared",
        f"  ...in its own top {scored['n_top']:<3}      "
        f"{scored['spearman_top_n']:+.3f}  <- the pool every pick comes from",
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
