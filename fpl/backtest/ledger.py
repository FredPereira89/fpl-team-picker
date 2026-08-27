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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .gw_level import evaluate_predictions

LEDGER_DIR = "predictions"


def save_predictions(xp: pd.DataFrame, gw: int, root: Path) -> Path:
    """Write one gameweek's xP frame. Overwrites: a re-run before the deadline
    supersedes the earlier forecast rather than accumulating drafts."""
    out = Path(root) / LEDGER_DIR
    out.mkdir(parents=True, exist_ok=True)
    frame = xp.copy()
    frame.insert(0, "gw", int(gw))
    path = out / f"gw{int(gw)}.parquet"
    frame.to_parquet(path, index=False)
    return path


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


def score_gameweek(pred: pd.DataFrame, actuals: pd.DataFrame,
                   pred_col: str = "xp_next1") -> dict:
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
    rho_played = 0.0
    if len(played) >= 3 and played[pred_col].nunique() > 1 and played["actual"].nunique() > 1:
        r = spearmanr(played[pred_col].values, played["actual"].values).statistic
        rho_played = 0.0 if np.isnan(r) else float(r)

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
        "spearman_played": rho_played,
        "n_played": int(len(played)),
    }


def scored_summary(scored: dict, gw: int) -> str:
    """One-screen verdict, written to be read by someone deciding whether to
    trust this week's recommendation."""
    by_pos = scored["spearman_by_position"]
    bias = scored["bias_by_position"]
    lines = [
        f"GW{gw} forecast scored against {scored['n']} players "
        f"({scored['n_played']} of them appeared):",
        f"  rank quality (Spearman)  {scored['spearman_overall']:+.3f} overall, "
        f"{scored['spearman_played']:+.3f} among players who appeared",
        f"  error                    MAE {scored['mae']:.2f}, RMSE {scored['rmse']:.2f}, "
        f"bias {scored['bias']:+.2f} pts/player",
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
        lines.append(
            "  WARNING: over-predicting " + ", ".join(sorted(over)) +
            " by more than 0.4 pts/player — the optimizer is buying those "
            "positions at inflated prices."
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
