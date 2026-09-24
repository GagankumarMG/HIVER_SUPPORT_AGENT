"""Exploratory intent discovery via clustering on customer messages.

Embeds cleaned customer texts (SentenceTransformer if available, else TF-IDF
fallback so this still runs with zero external downloads/GPU), clusters with
KMeans, and surfaces representative examples per cluster so a human can name
each cluster meaningfully in configs/intents.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DiscoveryResult:
    cluster_labels: np.ndarray
    representative_examples: dict[int, list[str]]
    silhouette: float
    method: str


def _embed_semantic(texts: list[str]) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:  # noqa: BLE001 - broad by design, this is an optional path
        logger.warning("Semantic embedding unavailable (%s); falling back to TF-IDF.", e)
        return None


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    return vec.fit_transform(texts).toarray()


def discover_intents(
    texts: list[str], n_clusters: int = 8, seed: int = 42, n_examples: int = 5
) -> DiscoveryResult:
    embeddings = _embed_semantic(texts)
    method = "semantic"
    if embeddings is None:
        embeddings = _embed_tfidf(texts)
        method = "tfidf"

    n_clusters = min(n_clusters, max(2, len(texts) // 5))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    try:
        sil = float(silhouette_score(embeddings, labels)) if n_clusters > 1 else 0.0
    except ValueError:
        sil = 0.0

    reps: dict[int, list[str]] = {}
    centers = km.cluster_centers_
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        dists = np.linalg.norm(embeddings[idx] - centers[c], axis=1)
        closest = idx[np.argsort(dists)[:n_examples]]
        reps[c] = [texts[i] for i in closest]

    logger.info(
        "Discovered %d clusters (method=%s, silhouette=%.3f)", n_clusters, method, sil
    )
    return DiscoveryResult(
        cluster_labels=labels,
        representative_examples=reps,
        silhouette=sil,
        method=method,
    )


def discovery_to_dataframe(texts: list[str], result: DiscoveryResult) -> pd.DataFrame:
    return pd.DataFrame({"text": texts, "cluster": result.cluster_labels})
