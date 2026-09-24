"""Dataset inspection: prints and saves summary statistics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def summarize(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    inbound_count = int(df["inbound"].sum())
    outbound_count = n_rows - inbound_count

    date_col = "created_at_parsed" if "created_at_parsed" in df.columns else None
    if date_col:
        valid_dates = df[date_col].dropna()
        date_range = {
            "min": str(valid_dates.min()) if len(valid_dates) else None,
            "max": str(valid_dates.max()) if len(valid_dates) else None,
        }
    else:
        date_range = {"min": None, "max": None}

    summary = {
        "n_rows": n_rows,
        "columns": list(df.columns),
        "missing_value_counts": {
            c: int(df[c].isna().sum()) for c in df.columns
        },
        "duplicate_row_count": int(df.duplicated().sum()),
        "duplicate_tweet_id_count": int(df["tweet_id"].duplicated().sum())
        if "tweet_id" in df.columns
        else None,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "date_range": date_range,
        "unique_authors": int(df["author_id"].nunique())
        if "author_id" in df.columns
        else None,
    }
    return summary


def save_summary(summary: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved dataset summary to %s", out_path)


def print_summary(summary: dict) -> None:
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Rows:                 {summary['n_rows']:,}")
    print(f"Columns:              {summary['columns']}")
    print(f"Duplicate rows:       {summary['duplicate_row_count']:,}")
    print(f"Duplicate tweet_ids:  {summary['duplicate_tweet_id_count']}")
    print(f"Inbound (customer):   {summary['inbound_count']:,}")
    print(f"Outbound (support):   {summary['outbound_count']:,}")
    print(f"Date range:           {summary['date_range']['min']} -> {summary['date_range']['max']}")
    print(f"Unique authors:       {summary['unique_authors']:,}")
    print("Missing values per column:")
    for col, n in summary["missing_value_counts"].items():
        print(f"  {col}: {n}")
    print("=" * 60)
