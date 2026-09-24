#!/usr/bin/env python
"""Quality-control checks on the labeled golden set. Fails loudly (non-zero
exit code) if the golden set is incomplete or invalid.

Usage:
    python scripts/check_golden_set.py
"""
from __future__ import annotations

import sys

import pandas as pd
import yaml

from src.escalation import reason_codes as rc
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VALID_ESCALATE_VALUES = {"AUTO_HANDLE", "ESCALATE"}


def main() -> int:
    path = "data/golden/golden_set.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FAIL: {path} not found. Run scripts/04_create_golden_candidates.py "
              "then annotate via `streamlit run app/streamlit_app.py`, saving the "
              "result as data/golden/golden_set.csv.")
        return 1

    errors: list[str] = []

    if df["example_id"].duplicated().any():
        errors.append("Duplicate example_id values found.")
    if df["customer_text"].duplicated().any():
        errors.append("Duplicate customer_text values found (near-duplicates allowed, exact not).")

    n_total = len(df)
    if not (150 <= n_total <= 250):
        errors.append(f"Golden set has {n_total} examples; must be within 150-250.")

    required_cols = ["gold_intent", "gold_escalate"]
    for col in required_cols:
        n_missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if n_missing:
            errors.append(f"{n_missing}/{n_total} rows missing required field '{col}'.")

    try:
        with open("configs/intents.yaml", "r", encoding="utf-8") as f:
            intents_cfg = yaml.safe_load(f)
        valid_intents = {i["name"] for i in intents_cfg.get("intents", [])}
    except FileNotFoundError:
        valid_intents = set()

    if valid_intents:
        bad_intents = set(df["gold_intent"].dropna().unique()) - valid_intents
        if bad_intents:
            errors.append(f"gold_intent contains values not in configs/intents.yaml: {bad_intents}")

    bad_escalate = set(df["gold_escalate"].dropna().unique()) - VALID_ESCALATE_VALUES
    if bad_escalate:
        errors.append(f"gold_escalate contains invalid values: {bad_escalate}")

    escalate_rows = df[df["gold_escalate"] == "ESCALATE"]
    missing_reason = escalate_rows["gold_escalation_reason"].isna() | (
        escalate_rows["gold_escalation_reason"].astype(str).str.strip() == ""
    )
    if missing_reason.any():
        errors.append(f"{missing_reason.sum()} ESCALATE rows missing gold_escalation_reason.")
    bad_reason_codes = set(
        escalate_rows.loc[~missing_reason, "gold_escalation_reason"].unique()
    ) - set(rc.ALL_REASON_CODES)
    if bad_reason_codes:
        errors.append(f"gold_escalation_reason contains unknown reason codes: {bad_reason_codes}")

    intent_counts = df["gold_intent"].value_counts()
    sparse = intent_counts[intent_counts < 3]
    if not sparse.empty:
        errors.append(
            f"WARNING: these intents have <3 examples in the golden set, "
            f"which may make per-intent metrics unstable: {sparse.to_dict()}"
        )

    completion_pct = 100.0 * (1 - df["gold_intent"].isna().mean())
    print(f"Golden set completion: {completion_pct:.1f}% ({n_total} rows)")

    if errors:
        print("FAIL: golden set quality checks did not pass:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PASS: golden set quality checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
