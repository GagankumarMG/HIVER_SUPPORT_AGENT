"""Lexical (BM25) retrieval over historical customer messages.

Uses the `rank_bm25` package when available. If it is not installed, falls
back to a small self-contained BM25-Okapi implementation (same scoring
formula, pure numpy) so retrieval keeps working in minimal environments
without silently degrading to a weaker method.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    from rank_bm25 import BM25Okapi as _ExternalBM25Okapi
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _ExternalBM25Okapi = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class _MinimalBM25Okapi:
    """Self-contained BM25-Okapi, used only as a fallback when the
    `rank_bm25` package isn't installed. Implements the standard Okapi BM25
    scoring formula (k1=1.5, b=0.75)."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if tokenized_corpus else 0.0
        self.doc_freqs = [Counter(d) for d in tokenized_corpus]
        df = Counter()
        for d in tokenized_corpus:
            for term in set(d):
                df[term] += 1
        n_docs = len(tokenized_corpus)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lens)):
            score = 0.0
            for term in query_tokens:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                f = freqs[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (f * (self.k1 + 1)) / (denom or 1)
            scores[i] = score
        return scores


@dataclass
class LexicalRetriever:
    corpus_texts: list[str]

    def __post_init__(self):
        self._tokenized = [_tokenize(t) for t in self.corpus_texts]
        if _ExternalBM25Okapi is not None:
            self.bm25 = _ExternalBM25Okapi(self._tokenized)
        else:
            self.bm25 = _MinimalBM25Okapi(self._tokenized)

    def score(self, query: str) -> np.ndarray:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=float)
        if scores.max() > 0:
            scores = scores / scores.max()  # normalize to [0,1] for hybrid combination
        return scores

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.score(query)
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]
