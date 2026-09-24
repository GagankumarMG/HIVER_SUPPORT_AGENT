#!/usr/bin/env python
"""Automated data-leakage checks (Section 39 of the assignment spec).

Verifies:
  - golden examples do not occur in the training split
  - exact duplicate customer texts do not cross splits
  - retrieval index (built from train.parquet) does not contain golden texts
  - evaluation labels (gold_*) were not generated from model predictions
    (structural check: gold columns must exist independently of any
    pred_ columns in golden_set.csv)

Usage:
    python scripts/12_leakage_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def main():
    result = {"checks": {}, "status": "PENDING"}

    golden_path = Path("data/golden/golden_set.csv")
    train_path = Path("data/processed/train.parquet")
    val_path = Path("data/processed/val.parquet")
    test_path = Path("data/processed/test.parquet")
    index_path = Path("models/hybrid_retrieval_index.joblib")

    if not golden_path.exists():
        result["status"] = "PENDING_GOLDEN_SET_ANNOTATION"
        _save(result)
        print(json.dumps(result, indent=2))
        return

    golden = pd.read_csv(golden_path)
    golden_texts = set(golden["customer_text"].dropna().tolist())

    if train_path.exists():
        train = load_table(train_path)
        overlap = set(train["cleaned_text"].dropna().tolist()) & golden_texts
        result["checks"]["golden_not_in_train"] = {
            "passed": len(overlap) == 0,
            "n_overlap": len(overlap),
        }
    else:
        result["checks"]["golden_not_in_train"] = {"passed": None, "note": "train.parquet not found"}

    if train_path.exists() and val_path.exists() and test_path.exists():
        train = load_table(train_path)
        val = load_table(val_path)
        test = load_table(test_path)
        t_texts = set(train["cleaned_text"])
        v_texts = set(val["cleaned_text"])
        te_texts = set(test["cleaned_text"])
        cross_overlap = (t_texts & v_texts) | (t_texts & te_texts) | (v_texts & te_texts)
        result["checks"]["no_cross_split_duplicates"] = {
            "passed": len(cross_overlap) == 0,
            "n_overlap": len(cross_overlap),
        }
    else:
        result["checks"]["no_cross_split_duplicates"] = {"passed": None, "note": "splits not found"}

    if index_path.exists():
        retriever = joblib.load(index_path)
        index_texts = set(retriever.records["customer_text"].dropna().tolist())
        overlap = index_texts & golden_texts
        result["checks"]["golden_not_in_retrieval_index"] = {
            "passed": len(overlap) == 0,
            "n_overlap": len(overlap),
        }
    else:
        result["checks"]["golden_not_in_retrieval_index"] = {"passed": None, "note": "index not built yet"}

    pred_cols_in_golden = [c for c in golden.columns if c.startswith("pred_")]
    result["checks"]["golden_labels_not_derived_from_predictions"] = {
        "passed": len(pred_cols_in_golden) == 0,
        "note": "golden_set.csv must only contain gold_* human labels, no pred_* columns",
    }

    all_passed = all(
        c.get("passed") in (True, None) for c in result["checks"].values()
    )
    any_failed = any(c.get("passed") is False for c in result["checks"].values())
    result["status"] = "FAIL" if any_failed else ("PASS" if all_passed else "PARTIAL_PENDING")

    _save(result)
    print(json.dumps(result, indent=2))


def _save(result: dict) -> None:
    with open("outputs/metrics/leakage_check.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
