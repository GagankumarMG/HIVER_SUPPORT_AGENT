"""Chronological train/validation/test split, leakage-safe.

The golden evaluation set (data/golden/golden_set.csv) is ALWAYS held out
entirely from these splits: golden examples are removed before splitting and
must never be used for classifier training or as retrieval-index documents.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    golden_customer_texts: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split by time order: oldest -> train, middle -> val, newest -> test.

    Any row whose customer_text exactly matches a golden-set example is
    dropped before splitting (leakage guard - see src/evaluation for the
    automated check that verifies this held).
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if golden_customer_texts:
        before = len(df)
        df = df[~df["customer_text"].isin(golden_customer_texts)].reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d rows overlapping with golden set", removed)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val :].reset_index(drop=True),
    }
    for name, part in splits.items():
        logger.info("Split %s: %d rows", name, len(part))
    return splits
