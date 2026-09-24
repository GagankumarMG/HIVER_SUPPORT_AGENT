#!/usr/bin/env python
"""Build the hybrid retrieval index from the TRAIN split only (never the
golden set, never val/test) and save it for reuse by the agent + baselines.

Usage:
    python scripts/06_build_index.py
"""
from __future__ import annotations

import pandas as pd

from src.retrieval.build_index import build_and_save_index
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def main():
    settings = load_settings()
    train = load_table("data/processed/train.parquet")
    corpus = train.rename(columns={"support_text": "support_reply"})[
        ["cleaned_text", "support_reply", "intent"]
    ].rename(columns={"cleaned_text": "customer_text"})
    corpus = corpus.dropna(subset=["customer_text", "support_reply"]).reset_index(drop=True)

    build_and_save_index(
        corpus,
        "models/hybrid_retrieval_index.joblib",
        lexical_weight=settings.get("retrieval", "lexical_weight", default=0.4),
        semantic_weight=settings.get("retrieval", "semantic_weight", default=0.6),
        embedding_model_name=settings.get(
            "intents", "embedding_model",
            default="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    print(f"Built retrieval index over {len(corpus)} historical TRAIN-split pairs.")


if __name__ == "__main__":
    main()
