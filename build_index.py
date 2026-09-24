"""Build and persist the retrieval corpus (index) from training-split
historical support pairs. Golden-set examples must be excluded (enforced by
the caller passing a corpus DataFrame already filtered to exclude them; see
scripts/06_build_index.py and src/evaluation/human_agreement.py for the
leakage checks that verify no overlap)."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_and_save_index(
    corpus_df: pd.DataFrame,
    out_path: str | Path,
    lexical_weight: float = 0.4,
    semantic_weight: float = 0.6,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HybridRetriever:
    retriever = HybridRetriever(
        corpus_df,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        embedding_model_name=embedding_model_name,
    )
    joblib.dump(retriever, out_path)
    logger.info("Saved retrieval index (%d records) to %s", len(corpus_df), out_path)
    return retriever


def load_index(path: str | Path) -> HybridRetriever:
    return joblib.load(path)
