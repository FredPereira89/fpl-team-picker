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
