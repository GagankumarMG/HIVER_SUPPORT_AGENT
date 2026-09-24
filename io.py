"""Tabular I/O helpers that prefer parquet (compact, preserves dtypes) but
transparently fall back to CSV if pyarrow/fastparquet are unavailable in the
runtime environment, so the pipeline degrades gracefully rather than
crashing on missing optional dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as e:
        csv_path = path.with_suffix(".csv")
        logger.warning(
            "Parquet engine unavailable (%s); saving %s as CSV instead.", e, csv_path
        )
        df.to_csv(csv_path, index=False)
        return csv_path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, FileNotFoundError) as e:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            logger.warning(
                "Parquet read failed (%s); loading fallback CSV %s instead.", e, csv_path
            )
            return pd.read_csv(csv_path)
        raise
