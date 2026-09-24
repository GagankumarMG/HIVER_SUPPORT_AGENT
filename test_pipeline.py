import pandas as pd
import pytest

from src.agent.pipeline import PipelineConfig, SupportAgentPipeline
from src.intents.classifier import TfidfLogisticClassifier
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def trained_pipeline():
    texts = [
        "my payment failed", "I was charged twice for my order",
        "please cancel my subscription", "I want to cancel my plan",
        "cannot log into my account", "forgot my password please help",
    ] * 3
    labels = (
        ["PAYMENT_ISSUE"] * 2 + ["CANCELLATION"] * 2 + ["LOGIN_OR_ACCOUNT"] * 2
    ) * 3
    clf = TfidfLogisticClassifier()
    clf.fit(texts, labels)

    records = pd.DataFrame(
        {
            "customer_text": texts,
            "support_reply": [f"Thanks, we're handling: {t}" for t in texts],
            "intent": labels,
        }
    )
    retriever = HybridRetriever(records)
    config = PipelineConfig(brand="TestBrand", confidence_threshold=0.0,
                             min_hybrid_score_for_grounding=0.0)
    return SupportAgentPipeline(clf, retriever, config, llm_client=None)


def test_pipeline_end_to_end_produces_valid_response(trained_pipeline):
    result = trained_pipeline.run("my payment failed again this month")
    assert result.intent in {"PAYMENT_ISSUE", "CANCELLATION", "LOGIN_OR_ACCOUNT"}
    assert result.decision in {"AUTO_HANDLE", "ESCALATE"}
    assert isinstance(result.draft_reply, str) and len(result.draft_reply) > 0
    assert len(result.retrieved_examples) > 0


def test_pipeline_escalates_with_no_retrieval_evidence():
    from src.retrieval.hybrid_retriever import HybridRetriever as HR

    records = pd.DataFrame(
        {
            "customer_text": ["totally unrelated historical topic xyz"],
            "support_reply": ["some unrelated reply"],
            "intent": ["OTHER_OR_UNKNOWN"],
        }
    )
    clf = TfidfLogisticClassifier()
    clf.fit(["totally unrelated historical topic xyz"] * 3, ["OTHER_OR_UNKNOWN"] * 3)
    retriever = HR(records)
    config = PipelineConfig(brand="TestBrand", confidence_threshold=0.0,
                             min_hybrid_score_for_grounding=0.99)
    pipeline = SupportAgentPipeline(clf, retriever, config, llm_client=None)
    result = pipeline.run("a completely different unrelated message about weather")
    assert result.decision == "ESCALATE"
