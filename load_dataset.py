"""Robust loading of the Kaggle "Customer Support on Twitter" CSV.

The dataset is large (~2.8M rows in the original Kaggle release), so callers
should avoid re-loading the full file repeatedly. Use `load_raw(nrows=...)`
during development.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

DTYPES = {
    "tweet_id": str,
    "author_id": str,
    "text": str,
    "response_tweet_id": str,
    "in_response_to_tweet_id": str,
}


class SchemaError(ValueError):
    """Raised when the input CSV does not match the expected schema."""


def find_csv(data_path: str | Path) -> Path:
    """Locate the dataset CSV. Accepts a direct file path or a directory,
    in which case it looks for the first *.csv file."""
    p = Path(data_path)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found under directory {p}")
        return candidates[0]
    raise FileNotFoundError(
        f"Dataset not found at {p}. Download it from "
        "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
        "and set DATA_PATH in .env, or place it at data/raw/twcs.csv. "
        "For a quick runnable demo, use the bundled synthetic dataset via "
        "scripts/00b_make_synthetic_dataset.py."
    )


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            "This does not match the expected Kaggle 'Customer Support on "
            "Twitter' schema (tweet_id, author_id, inbound, created_at, text, "
            "response_tweet_id, in_response_to_tweet_id)."
        )


def load_raw(data_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load and lightly validate the raw dataset."""
    csv_path = find_csv(data_path)
    logger.info("Loading raw dataset from %s (nrows=%s)", csv_path, nrows)
    df = pd.read_csv(csv_path, dtype=DTYPES, nrows=nrows, low_memory=False)
    validate_schema(df)
    # inbound arrives as True/False or "True"/"False" strings depending on
    # pandas version / source export; normalize to bool.
    if df["inbound"].dtype != bool:
        df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True
    )
    n_bad_dates = df["created_at_parsed"].isna().sum()
    if n_bad_dates:
        logger.warning("%d rows had unparseable created_at timestamps", n_bad_dates)
    return df
