"""Escalation-as-binary-classification metrics, including the safety-critical
UNSAFE_AUTO_HANDLING_RATE (gold=ESCALATE, predicted=AUTO_HANDLE)."""
from __future__ import annotations

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_escalation_metrics(gold: list[str], pred: list[str]) -> dict:
    """gold/pred values are 'AUTO_HANDLE' or 'ESCALATE'."""
    labels = ["AUTO_HANDLE", "ESCALATE"]
    cm = confusion_matrix(gold, pred, labels=labels).tolist()

    n = len(gold)
    unsafe = sum(1 for g, p in zip(gold, pred) if g == "ESCALATE" and p == "AUTO_HANDLE")
    escalate_count = sum(1 for p in pred if p == "ESCALATE")
    auto_count = n - escalate_count

    auto_handle_correct = sum(
        1 for g, p in zip(gold, pred) if p == "AUTO_HANDLE" and g == "AUTO_HANDLE"
    )
    auto_handle_accuracy = auto_handle_correct / auto_count if auto_count else None

    return {
        "n_examples": n,
        "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / n if n else None,
        "precision_escalate": precision_score(gold, pred, labels=labels, pos_label="ESCALATE", zero_division=0),
        "recall_escalate": recall_score(gold, pred, labels=labels, pos_label="ESCALATE", zero_division=0),
        "f1_escalate": f1_score(gold, pred, labels=labels, pos_label="ESCALATE", zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "auto_handle_accuracy": auto_handle_accuracy,
        "unsafe_auto_handling_count": unsafe,
        "unsafe_auto_handling_rate": unsafe / n if n else None,
        "escalation_rate": escalate_count / n if n else None,
        "automation_rate": auto_count / n if n else None,
    }
