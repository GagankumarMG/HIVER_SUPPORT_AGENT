"""Prompt templates for grounded reply generation."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a customer support drafting assistant. Draft a response using "
    "only information supported by the customer message and the retrieved "
    "historical support examples. Do not invent refunds, prices, timelines, "
    "policies, account details, compensation, links, or promises. If the "
    "historical examples do not provide enough evidence, recommend human "
    "escalation rather than guessing."
)

RESPONSE_JSON_SCHEMA_HINT = """
Return ONLY a JSON object with exactly this shape, no extra text:
{
  "reply": "string",
  "grounded": true or false,
  "confidence": 0.0-1.0,
  "unsupported_claims": ["string", ...],
  "recommended_action": "AUTO_HANDLE" or "ESCALATE",
  "escalation_reason": "string or null"
}
"""


def build_user_prompt(
    brand: str,
    intent: str,
    customer_message: str,
    conversation_context: str | None,
    historical_examples: list[dict],
) -> str:
    examples_block = "\n".join(
        f"- Historical customer message: {ex['historical_customer_message']}\n"
        f"  Historical support response: {ex['historical_support_response']}\n"
        f"  Retrieval score: {ex['retrieval_score']:.2f}"
        for ex in historical_examples
    ) or "(no sufficiently similar historical examples found)"

    return f"""Brand: {brand}
Detected intent: {intent}
Customer message: {customer_message}
Conversation context: {conversation_context or "(none)"}

Top historical examples for grounding:
{examples_block}

{RESPONSE_JSON_SCHEMA_HINT}
"""
