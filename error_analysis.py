"""Failure-mode extraction from real held-out evaluation data. Never invents
examples: only real (message, prediction, gold) rows already present in the
evaluation results DataFrame are used."""
from __future__ import annotations

import pandas as pd


def categorize_failure(row: pd.Series) -> str | None:
    """Assigns a failure category to a single evaluation row, or None if the
    row was actually correct/not a failure."""
    intent_correct = row.get("gold_intent") == row.get("pred_intent")
    escalation_correct = row.get("gold_escalate") == row.get("pred_decision")

    if intent_correct and escalation_correct:
        return None

    msg = str(row.get("customer_text", ""))
    if len(msg.split()) <= 4:
        return "VERY_SHORT_MESSAGE"
    if row.get("gold_intent") == "OTHER_OR_UNKNOWN" and row.get("pred_intent") != "OTHER_OR_UNKNOWN":
        return "NOVEL_OR_AMBIGUOUS_INTENT"
    if row.get("gold_escalate") == "ESCALATE" and row.get("pred_decision") == "AUTO_HANDLE":
        return "UNSAFE_AUTO_HANDLING"
    if not intent_correct:
        return "INTENT_MISCLASSIFICATION"
    if not escalation_correct:
        return "ESCALATION_THRESHOLD_ERROR"
    return "OTHER_FAILURE"


def extract_failure_modes(eval_df: pd.DataFrame, top_n: int = 5, examples_per_mode: int = 3) -> pd.DataFrame:
    """Returns a DataFrame with columns: failure_category, n_examples, and
    up to `examples_per_mode` representative REAL rows per category (as JSON
    strings) for docs/report inclusion."""
    df = eval_df.copy()
    df["failure_category"] = df.apply(categorize_failure, axis=1)
    failures = df[df["failure_category"].notna()]
    if failures.empty:
        return pd.DataFrame(
            columns=["failure_category", "n_examples", "example_customer_text",
                     "pred_intent", "gold_intent", "pred_decision", "gold_escalate"]
        )

    counts = failures["failure_category"].value_counts().head(top_n)
    rows = []
    for category, n in counts.items():
        subset = failures[failures["failure_category"] == category].head(examples_per_mode)
        for _, r in subset.iterrows():
            rows.append(
                {
                    "failure_category": category,
                    "n_examples": int(n),
                    "example_customer_text": r.get("customer_text"),
                    "pred_intent": r.get("pred_intent"),
                    "gold_intent": r.get("gold_intent"),
                    "pred_decision": r.get("pred_decision"),
                    "gold_escalate": r.get("gold_escalate"),
                    "retrieved_top_example": r.get("retrieved_top_example"),
                }
            )
    return pd.DataFrame(rows)
