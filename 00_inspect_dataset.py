#!/usr/bin/env python
"""Inspect the raw dataset and save a schema/statistics summary.

Usage:
    python scripts/00_inspect_dataset.py
"""
from __future__ import annotations

from src.data.inspect_schema import print_summary, save_summary, summarize
from src.data.load_dataset import load_raw
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    settings = load_settings()
    df = load_raw(settings.data_path)
    summary = summarize(df)
    print_summary(summary)
    save_summary(summary, "outputs/metrics/dataset_summary.json")


if __name__ == "__main__":
    main()
