import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_retriever import LexicalRetriever


def test_lexical_retriever_top_k_ranks_relevant_first():
    corpus = [
        "my payment failed and I was charged",
        "what is your return policy",
        "the weather is nice today",
    ]
    retriever = LexicalRetriever(corpus)
    results = retriever.top_k("payment charged twice", k=2)
    assert results[0][0] == 0  # most relevant doc index


def test_hybrid_retriever_returns_requested_k():
    records = pd.DataFrame(
        {
            "customer_text": [
                "payment failed for order 123",
                "cancel my subscription please",
                "app keeps crashing on login",
            ],
            "support_reply": [
                "we've flagged the payment issue",
                "subscription cancelled",
                "please try reinstalling the app",
            ],
            "intent": ["PAYMENT_ISSUE", "CANCELLATION", "TECHNICAL_PROBLEM"],
        }
    )
    retriever = HybridRetriever(records, lexical_weight=0.4, semantic_weight=0.6)
    results = retriever.retrieve("my payment did not go through", k=2)
    assert len(results) == 2
    assert all(0.0 <= r.retrieval_score <= 1.0 for r in results)
