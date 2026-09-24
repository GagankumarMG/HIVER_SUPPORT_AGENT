#!/usr/bin/env python
"""Extract top-5 real failure modes from the main system's golden-set
evaluation rows (outputs/examples/main_system_eval_rows.csv).

Usage:
    python scripts/10_analyze_failures.py
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.error_analysis import extract_failure_modes
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    path = "outputs/examples/main_system_eval_rows.csv"
    try:
        eval_df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{path} not found. Run scripts/08_evaluate.py first.")
        return

    failures = extract_failure_modes(eval_df, top_n=5, examples_per_mode=3)
    out_path = "outputs/examples/failure_analysis.csv"
    failures.to_csv(out_path, index=False)
    logger.info("Saved failure analysis (%d rows) to %s", len(failures), out_path)

    if failures.empty:
        print("No failures found in the golden-set evaluation (all predictions "
              "matched gold labels) -- unusual; double check the golden set is "
              "non-trivial and the pipeline is actually being exercised.")
        return

    print(failures.groupby("failure_category")["n_examples"].first().sort_values(ascending=False))


if __name__ == "__main__":
    main()
