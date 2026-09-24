"""Semantic (embedding cosine-similarity) retrieval over historical customer
messages. Falls back to TF-IDF vectors if sentence-transformers is
unavailable, so retrieval keeps working offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticRetriever:
    corpus_texts: list[str]
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings: np.ndarray = field(init=False, repr=False)
    _tfidf: TfidfVectorizer | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.embeddings = self._build_embeddings(self.corpus_texts)

    def _build_embeddings(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model_name)
            return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SentenceTransformer unavailable (%s); semantic retrieval falling "
                "back to TF-IDF cosine similarity.", e
            )
            self._tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            return self._tfidf.fit_transform(texts).toarray()

    def _embed_query(self, query: str) -> np.ndarray:
        if self._tfidf is not None:
            return self._tfidf.transform([query]).toarray()
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model_name)
            return np.asarray(model.encode([query], show_progress_bar=False, normalize_embeddings=True))
        except Exception:  # noqa: BLE001
            # Should not happen if corpus used TF-IDF fallback consistently.
            raise RuntimeError("Semantic retriever misconfigured: no embedder available for query.")

    def score(self, query: str) -> np.ndarray:
        q = self._embed_query(query)
        sims = cosine_similarity(q, self.embeddings)[0]
        sims = np.clip(sims, 0, 1)
        return sims

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.score(query)
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]
