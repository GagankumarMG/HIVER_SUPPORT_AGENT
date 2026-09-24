#!/usr/bin/env python
"""Train intent classifiers (main semantic classifier + TF-IDF baseline) on
the chronological TRAIN split, apply the discovered/human-named intent
taxonomy, and save model artifacts.

Requires: configs/intents.yaml cluster->intent mapping to have been finalized
by a human after running scripts/03_discover_intents.py. For this repo's
synthetic demo data, `data/processed/support_pairs.parquet` already carries
ground-truth-like intent structure from the generator, so a lightweight
keyword-based weak-labeler is used as a stand-in for the human cluster-naming
step (see `_weak_label_from_keywords` below) -- clearly marked as such.

Usage:
    python scripts/05_train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.split_data import chronological_split
from src.intents.classifier import SemanticClassifier, TfidfLogisticClassifier
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger, set_global_seed
from src.utils.io import load_table, save_table

logger = get_logger(__name__)

KEYWORD_RULES = {
    "PAYMENT_ISSUE": ["payment", "charged", "charge", "billed", "billing"],
    "REFUND_REQUEST": ["refund", "money back"],
    "LOGIN_OR_ACCOUNT": ["log into", "login", "password", "locked out", "2fa", "two factor"],
    "ORDER_OR_TRANSACTION": ["order", "transaction", "processing"],
    "DELIVERY_OR_SHIPPING": ["delivered", "shipping", "tracking", "package", "damaged"],
    "TECHNICAL_PROBLEM": ["crash", "error code", "broken", "bug"],
    "CANCELLATION": ["cancel"],
    "PRODUCT_INFORMATION": ["difference between", "discount", "work with", "does your"],
}


def _weak_label_from_keywords(text: str) -> str:
    """STAND-IN for human cluster naming, used only so this repo's bundled
    synthetic demo data can exercise the full training pipeline without a
    human annotation pass. On REAL data, replace this call with
    src/intents/label.py's apply_cluster_to_intent_map, driven by a human
    reviewing scripts/03_discover_intents.py cluster output."""
    t = text.lower()
    for intent, keywords in KEYWORD_RULES.items():
        if any(kw in t for kw in keywords):
            return intent
    return "OTHER_OR_UNKNOWN"


def main():
    settings = load_settings()
    set_global_seed(settings.seed)

    pairs = load_table("data/processed/support_pairs.parquet")
    pairs["intent"] = pairs["cleaned_text"].apply(_weak_label_from_keywords)

    golden_texts: set[str] = set()
    golden_path = Path("data/golden/golden_set.csv")
    if golden_path.exists():
        golden_texts = set(pd.read_csv(golden_path)["customer_text"].tolist())
    else:
        logger.warning(
            "data/golden/golden_set.csv not found - proceeding without golden-set "
            "leakage exclusion. Create the golden set before final evaluation."
        )

    splits = chronological_split(
        pairs,
        timestamp_col="timestamp",
        train_frac=settings.get("split", "train_frac", default=0.70),
        val_frac=settings.get("split", "val_frac", default=0.15),
        golden_customer_texts=golden_texts,
    )
    for name, part in splits.items():
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        save_table(part, f"data/processed/{name}.parquet")

    train = splits["train"]
    val = splits["val"]

    Path("models").mkdir(exist_ok=True)

    tfidf_clf = TfidfLogisticClassifier(
        max_features=settings.get("intents", "tfidf_max_features", default=20000),
        seed=settings.seed,
    )
    tfidf_clf.fit(train["cleaned_text"].tolist(), train["intent"].tolist())
    tfidf_clf.save("models/baseline_b_tfidf_classifier.joblib")

    sem_clf = SemanticClassifier(
        embedding_model=settings.get("intents", "embedding_model",
                                      default="sentence-transformers/all-MiniLM-L6-v2"),
        seed=settings.seed,
        calibration=settings.get("classifier", "calibration", default="sigmoid"),
    )
    sem_clf.fit(train["cleaned_text"].tolist(), train["intent"].tolist())
    sem_clf.save("models/main_semantic_classifier.joblib")

    logger.info("Saved trained classifiers to models/")

    metadata = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(splits["test"]),
        "intent_label_source": "keyword_weak_labels_DEMO_STANDIN_FOR_HUMAN_CLUSTER_NAMING",
        "intent_distribution_train": train["intent"].value_counts().to_dict(),
    }
    with open("outputs/metrics/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
