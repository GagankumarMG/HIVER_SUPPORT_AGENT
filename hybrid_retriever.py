"""Hybrid retrieval combining lexical + semantic similarity.

Default weights: 0.4 lexical + 0.6 semantic (configurable via configs/config.yaml).
Never retrieves examples that are part of the golden evaluation set (the
corpus passed in at construction time must already exclude golden examples;
see scripts/06_build_index.py which enforces this).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.semantic_retriever import SemanticRetriever


@dataclass
class RetrievedExample:
    historical_customer_message: str
    historical_support_response: str
    intent: str | None
    retrieval_score: float
    intent_match: bool | None
    lexical_score: float
    semantic_score: float


class HybridRetriever:
    def __init__(
        self,
        records: pd.DataFrame,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """`records` must contain columns: customer_text, support_reply, intent
        (intent may be all-null if intents haven't been assigned yet)."""
        assert {"customer_text", "support_reply"}.issubset(records.columns)
        self.records = records.reset_index(drop=True)
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        texts = self.records["customer_text"].fillna("").tolist()
        self.lexical = LexicalRetriever(texts)
        self.semantic = SemanticRetriever(texts, embedding_model_name=embedding_model_name)

    def retrieve(
        self, query: str, k: int = 5, query_intent: str | None = None
    ) -> list[RetrievedExample]:
        lex_scores = self.lexical.score(query)
        sem_scores = self.semantic.score(query)
        hybrid = self.lexical_weight * lex_scores + self.semantic_weight * sem_scores

        order = hybrid.argsort()[::-1][:k]
        results = []
        for i in order:
            row = self.records.iloc[int(i)]
            intent_match = None
            if query_intent is not None and "intent" in row and pd.notna(row.get("intent")):
                intent_match = row["intent"] == query_intent
            results.append(
                RetrievedExample(
                    historical_customer_message=row["customer_text"],
                    historical_support_response=row["support_reply"],
                    intent=row.get("intent"),
                    retrieval_score=float(hybrid[i]),
                    intent_match=intent_match,
                    lexical_score=float(lex_scores[i]),
                    semantic_score=float(sem_scores[i]),
                )
            )
        return results
