"""End-to-end agent pipeline: preprocessing -> intent classification ->
{retrieval, escalation-signal-gathering} -> grounded reply generation ->
final escalation decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent.schemas import AgentResponse, RetrievedExampleOut, TopIntent
from src.data.preprocess import clean_text
from src.escalation.policy import decide_escalation
from src.generation.llm_client import LLMClient
from src.generation.reply_generator import generate_reply
from src.intents.classifier import IntentPrediction
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    brand: str
    confidence_threshold: float = 0.60
    min_hybrid_score_for_grounding: float = 0.35
    top_k_retrieval: int = 5
    high_risk_keywords: list[str] | None = None
    max_tokens: int = 400
    temperature: float = 0.2


class SupportAgentPipeline:
    def __init__(
        self,
        classifier,
        retriever: HybridRetriever,
        config: PipelineConfig,
        llm_client: LLMClient | None = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.config = config
        self.llm_client = llm_client

    def run(self, customer_message: str, conversation_context: str | None = None) -> AgentResponse:
        cleaned = clean_text(customer_message)

        pred: IntentPrediction = self.classifier.predict(
            cleaned, confidence_threshold=self.config.confidence_threshold
        )

        retrieved = self.retriever.retrieve(
            cleaned, k=self.config.top_k_retrieval, query_intent=pred.intent
        )

        gen = generate_reply(
            brand=self.config.brand,
            intent=pred.intent,
            customer_message=cleaned,
            conversation_context=conversation_context,
            retrieved=retrieved,
            llm_client=self.llm_client,
            min_hybrid_score_for_grounding=self.config.min_hybrid_score_for_grounding,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        top_score = retrieved[0].retrieval_score if retrieved else 0.0
        conflicting = _detect_conflicting_history(retrieved)

        escalation = decide_escalation(
            customer_message=cleaned,
            intent=pred.intent,
            intent_confidence=pred.confidence,
            confidence_threshold=self.config.confidence_threshold,
            retrieved_top_score=top_score,
            min_hybrid_score_for_grounding=self.config.min_hybrid_score_for_grounding,
            generation_grounded=gen.grounded,
            generation_unsupported_claims=gen.unsupported_claims,
            high_risk_keywords=self.config.high_risk_keywords or [],
            conflicting_history=conflicting,
        )

        return AgentResponse(
            customer_message=customer_message,
            intent=pred.intent,
            intent_confidence=pred.confidence,
            top_intents=[TopIntent(intent=i, score=s) for i, s in pred.top_k],
            retrieved_examples=[
                RetrievedExampleOut(
                    customer_message=r.historical_customer_message,
                    historical_reply=r.historical_support_response,
                    score=r.retrieval_score,
                )
                for r in retrieved
            ],
            draft_reply=gen.reply,
            grounded=gen.grounded,
            unsupported_claims=gen.unsupported_claims,
            decision=escalation.decision,
            escalation_reason=escalation.reason if escalation.decision == "ESCALATE" else None,
            reason_code=escalation.reason_code,
            generation_mode=gen.generation_mode,
        )


def _detect_conflicting_history(retrieved) -> bool:
    """Simple heuristic: if the top-2 retrieved historical replies have very
    low lexical overlap despite similar customer messages, flag as conflicting."""
    if len(retrieved) < 2:
        return False
    a, b = retrieved[0], retrieved[1]
    if abs(a.retrieval_score - b.retrieval_score) > 0.15:
        return False  # not really competing candidates
    tokens_a = set(a.historical_support_response.lower().split())
    tokens_b = set(b.historical_support_response.lower().split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return overlap < 0.05
