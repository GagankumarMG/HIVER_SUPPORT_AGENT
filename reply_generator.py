"""Grounded reply generation with an LLM-free fallback.

If OPENAI_API_KEY is unset (or the LLM call fails), the fallback:
  1. Uses the best retrieved historical response, minimally adapted.
  2. If no sufficiently similar historical answer exists (min_hybrid_score_for_grounding),
     returns a safe generic escalation message.
This keeps the CORE evaluation pipeline fully reproducible without any API key.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generation.llm_client import LLMClient, LLMUnavailableError
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.hybrid_retriever import RetrievedExample
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FALLBACK_NO_EVIDENCE_REPLY = (
    "I\u2019m sorry you\u2019re running into this issue. I\u2019d like to make sure you "
    "get the correct help, so this should be reviewed by our support team."
)


@dataclass
class GenerationResult:
    reply: str
    grounded: bool
    confidence: float
    unsupported_claims: list[str]
    recommended_action: str
    escalation_reason: str | None
    generation_mode: str  # "llm" or "fallback_retrieval" or "fallback_no_evidence"


def generate_fallback(
    retrieved: list[RetrievedExample], min_hybrid_score_for_grounding: float = 0.35
) -> GenerationResult:
    if not retrieved or retrieved[0].retrieval_score < min_hybrid_score_for_grounding:
        return GenerationResult(
            reply=FALLBACK_NO_EVIDENCE_REPLY,
            grounded=False,
            confidence=0.0,
            unsupported_claims=[],
            recommended_action="ESCALATE",
            escalation_reason="No sufficiently similar resolved historical interaction was found.",
            generation_mode="fallback_no_evidence",
        )
    best = retrieved[0]
    # Minimal, safe adaptation: use the historical response as-is. We do NOT
    # attempt to auto-substitute entities (order ids, names) since doing so
    # without an LLM risks fabricating unsupported specifics.
    return GenerationResult(
        reply=best.historical_support_response,
        grounded=True,
        confidence=best.retrieval_score,
        unsupported_claims=[],
        recommended_action="AUTO_HANDLE",
        escalation_reason=None,
        generation_mode="fallback_retrieval",
    )


def generate_reply(
    brand: str,
    intent: str,
    customer_message: str,
    conversation_context: str | None,
    retrieved: list[RetrievedExample],
    llm_client: LLMClient | None,
    min_hybrid_score_for_grounding: float = 0.35,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> GenerationResult:
    if llm_client is not None and llm_client.available:
        try:
            examples_payload = [
                {
                    "historical_customer_message": r.historical_customer_message,
                    "historical_support_response": r.historical_support_response,
                    "retrieval_score": r.retrieval_score,
                }
                for r in retrieved
            ]
            user_prompt = build_user_prompt(
                brand, intent, customer_message, conversation_context, examples_payload
            )
            parsed = llm_client.complete_json(
                SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, temperature=temperature
            )
            if not parsed.get("reply"):
                raise ValueError("LLM returned empty/invalid reply payload")
            return GenerationResult(
                reply=parsed["reply"],
                grounded=bool(parsed.get("grounded", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                unsupported_claims=list(parsed.get("unsupported_claims") or []),
                recommended_action=parsed.get("recommended_action", "ESCALATE"),
                escalation_reason=parsed.get("escalation_reason"),
                generation_mode="llm",
            )
        except (LLMUnavailableError, Exception) as e:  # noqa: BLE001
            logger.warning("LLM generation failed (%s); using LLM-free fallback.", e)

    return generate_fallback(retrieved, min_hybrid_score_for_grounding)
