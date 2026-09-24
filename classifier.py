"""Intent classifiers: TF-IDF+LogisticRegression baseline and the main
semantic (SentenceTransformer embeddings + calibrated LogisticRegression)
classifier. Both share a common interface so evaluation code can compare
them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    top_k: list[tuple[str, float]]


class IntentClassifier(Protocol):
    def fit(self, texts: list[str], labels: list[str]) -> "IntentClassifier": ...
    def predict(self, text: str, confidence_threshold: float = 0.0) -> IntentPrediction: ...
    def save(self, path: str | Path) -> None: ...


class TfidfLogisticClassifier:
    """Baseline B classifier: TF-IDF + Logistic Regression."""

    def __init__(self, max_features: int = 20000, seed: int = 42):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=1
        )
        self.model = LogisticRegression(max_iter=1000, random_state=seed)
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogisticClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, text: str, confidence_threshold: float = 0.0) -> IntentPrediction:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        top_k = [(self.classes_[i], float(proba[i])) for i in order[:3]]
        best_intent, best_conf = top_k[0]
        if best_conf < confidence_threshold:
            best_intent = "OTHER_OR_UNKNOWN"
        return IntentPrediction(best_intent, best_conf, top_k)

    def save(self, path: str | Path) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model,
                     "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfLogisticClassifier":
        obj = joblib.load(path)
        inst = cls()
        inst.vectorizer = obj["vectorizer"]
        inst.model = obj["model"]
        inst.classes_ = obj["classes"]
        return inst


class SemanticClassifier:
    """Main system classifier: SentenceTransformer embeddings + calibrated
    Logistic Regression. Falls back to TF-IDF embeddings automatically if
    sentence-transformers / model download is unavailable, so the pipeline
    stays runnable offline (see docs/decision_log.md).
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 seed: int = 42, calibration: str = "sigmoid"):
        self.embedding_model_name = embedding_model
        self._st_model = None
        self._tfidf_fallback: TfidfVectorizer | None = None
        base = LogisticRegression(max_iter=2000, random_state=seed)
        self.model = CalibratedClassifierCV(base, method=calibration, cv=3)
        self.classes_: list[str] = []
        self.seed = seed

    def _get_embedder(self):
        if self._st_model is not None or self._tfidf_fallback is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Loaded SentenceTransformer embedder: %s", self.embedding_model_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SentenceTransformer unavailable (%s). Falling back to TF-IDF "
                "embeddings for the 'semantic' classifier so the pipeline still runs.",
                e,
            )
            self._tfidf_fallback = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def _embed(self, texts: list[str], fit: bool = False) -> np.ndarray:
        self._get_embedder()
        if self._st_model is not None:
            return np.asarray(
                self._st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            )
        assert self._tfidf_fallback is not None
        if fit:
            return self._tfidf_fallback.fit_transform(texts).toarray()
        return self._tfidf_fallback.transform(texts).toarray()

    def fit(self, texts: list[str], labels: list[str]) -> "SemanticClassifier":
        X = self._embed(texts, fit=True)
        # CalibratedClassifierCV with cv=3 needs >= 3 examples per class; guard.
        min_class_count = min(labels.count(c) for c in set(labels))
        if min_class_count < 3:
            logger.warning(
                "Smallest class has only %d examples (<3); disabling CV calibration "
                "and fitting a plain LogisticRegression instead.", min_class_count
            )
            self.model = LogisticRegression(max_iter=2000, random_state=self.seed)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, text: str, confidence_threshold: float = 0.0) -> IntentPrediction:
        X = self._embed([text], fit=False)
        proba = self.model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        top_k = [(self.classes_[i], float(proba[i])) for i in order[:3]]
        best_intent, best_conf = top_k[0]
        if best_conf < confidence_threshold:
            best_intent = "OTHER_OR_UNKNOWN"
        return IntentPrediction(best_intent, best_conf, top_k)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "classes": self.classes_,
                "embedding_model_name": self.embedding_model_name,
                "used_tfidf_fallback": self._st_model is None,
                "tfidf_fallback": self._tfidf_fallback,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticClassifier":
        obj = joblib.load(path)
        inst = cls(embedding_model=obj["embedding_model_name"])
        inst.model = obj["model"]
        inst.classes_ = obj["classes"]
        if obj.get("used_tfidf_fallback"):
            inst._tfidf_fallback = obj["tfidf_fallback"]
        return inst
