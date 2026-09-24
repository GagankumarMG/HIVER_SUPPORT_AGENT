"""Baseline A (trivial) and Baseline B (simple/classical) implementations,
sharing the AgentResponse-like output shape for uniform evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.intents.classifier import TfidfLogisticClassifier
from src.retrieval.lexical_retriever import LexicalRetriever

GENERIC_FALLBACK_REPLY = (
    "Thanks for reaching out. We're sorry you're having trouble. Please share "
    "more details so our support team can help."
)


@dataclass
class BaselineResult:
    intent: str
    reply: str
    decision: str
    reason: str


class BaselineA:
    """Majority intent classifier + generic fallback reply + ALWAYS escalate."""

    def __init__(self):
        self.majority_intent: str | None = None

    def fit(self, labels: list[str]) -> "BaselineA":
        self.majority_intent = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, text: str) -> BaselineResult:
        return BaselineResult(
            intent=self.majority_intent or "OTHER_OR_UNKNOWN",
            reply=GENERIC_FALLBACK_REPLY,
            decision="ESCALATE",
            reason="Baseline A always escalates by design.",
        )


class BaselineB:
    """TF-IDF + LogisticRegression intent classifier; nearest-neighbor TF-IDF
    retrieval for reply; simple confidence-threshold escalation rule."""

    def __init__(self, confidence_threshold: float = 0.60):
        self.classifier = TfidfLogisticClassifier()
        self.confidence_threshold = confidence_threshold
        self.retriever: LexicalRetriever | None = None
        self.corpus: pd.DataFrame | None = None

    def fit(self, texts: list[str], labels: list[str], corpus: pd.DataFrame) -> "BaselineB":
        """`corpus` must have columns customer_text, support_reply (historical pairs
        used for nearest-neighbor reply retrieval; excludes the golden set)."""
        self.classifier.fit(texts, labels)
        self.corpus = corpus.reset_index(drop=True)
        self.retriever = LexicalRetriever(self.corpus["customer_text"].fillna("").tolist())
        return self

    def predict(self, text: str) -> BaselineResult:
        pred = self.classifier.predict(text, confidence_threshold=0.0)
        top = self.retriever.top_k(text, k=1)
        if top:
            idx, score = top[0]
            reply = self.corpus.iloc[idx]["support_reply"]
        else:
            reply = GENERIC_FALLBACK_REPLY
            score = 0.0
        decision = "AUTO_HANDLE" if pred.confidence >= self.confidence_threshold else "ESCALATE"
        reason = (
            f"Classifier confidence {pred.confidence:.2f} vs threshold {self.confidence_threshold:.2f}."
        )
        return BaselineResult(intent=pred.intent, reply=reply, decision=decision, reason=reason)
