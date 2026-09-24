"""Reply-quality proxy metrics that don't require an LLM judge or human
labels: historical similarity, semantic similarity to a retrieved reference,
and simple grounding heuristics. These SUPPLEMENT, never replace, the LLM
judge + human agreement evaluation required by the assignment (BLEU/ROUGE
are explicitly NOT used as the primary metric - see docs/decision_log.md)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def historical_similarity(reply: str, historical_reference: str) -> float:
    """TF-IDF cosine similarity between a generated reply and the best
    retrieved historical reply, as a cheap grounding proxy."""
    if not reply or not historical_reference:
        return 0.0
    vec = TfidfVectorizer().fit([reply, historical_reference])
    vecs = vec.transform([reply, historical_reference])
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])


def contains_unsupported_numeric_claim(reply: str, historical_reference: str) -> bool:
    """Heuristic grounding check: flags dollar amounts or day-counts in the
    reply that do not appear anywhere in the retrieved historical reference.
    This is intentionally simple/explainable per the assignment's engineering
    principles (avoid unnecessary complexity)."""
    import re

    reply_numbers = set(re.findall(r"\$\d+(?:\.\d+)?|\b\d+\s*(?:day|days|hour|hours)\b", reply.lower()))
    ref_numbers = set(re.findall(r"\$\d+(?:\.\d+)?|\b\d+\s*(?:day|days|hour|hours)\b", historical_reference.lower()))
    return bool(reply_numbers - ref_numbers)
