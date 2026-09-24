#!/usr/bin/env python
"""Load raw data and write a lightly-cleaned parquet for downstream steps.
This does NOT yet select a brand or reconstruct threads (see 02 and the
thread reconstruction call inside 02/03) - it's the generic ingestion step.

Usage:
    python scripts/01_prepare_data.py
"""
from __future__ import annotations

from pathlib import Path

from src.data.load_dataset import load_raw
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def main():
    settings = load_settings()
    df = load_raw(settings.data_path)
    out_path = Path("data/interim/raw_loaded.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_table(df, out_path)
    logger.info("Saved loaded raw dataset (%d rows) to %s", len(df), out_path)


if __name__ == "__main__":
    main()
