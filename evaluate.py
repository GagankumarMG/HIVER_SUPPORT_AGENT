"""Classification metrics: accuracy, macro/weighted F1, per-intent P/R/F1,
confusion matrix. Shared by baseline and main-system evaluation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_intent": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
        "n_examples": len(y_true),
        "labels": labels,
    }
    return metrics


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], out_path: str | Path) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Intent Classification Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)
