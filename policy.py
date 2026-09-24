"""Explicit escalation policy.

Escalates on: low classifier confidence, OTHER_OR_UNKNOWN intent,
insufficient retrieval evidence, no close historical resolution, sensitive
account actions / high-risk keywords, conflicting retrieved evidence,
generation-time grounding failure, or repeated unresolved interaction
signals. Designed to avoid escalating everything (see Baseline A, which
always escalates, as the contrast point in evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.escalation import reason_codes as rc


@dataclass
class EscalationDecision:
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason_code: str | None
    reason: str


SENSITIVE_ACCOUNT_KEYWORDS = [
    "delete my account",
    "close my account",
    "change my email",
    "change my password to",
    "transfer ownership",
]


def _has_high_risk_keyword(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    for kw in keywords:
        if kw.lower() in lowered:
            return kw
    return None


def decide_escalation(
    customer_message: str,
    intent: str,
    intent_confidence: float,
    confidence_threshold: float,
    retrieved_top_score: float,
    min_hybrid_score_for_grounding: float,
    generation_grounded: bool,
    generation_unsupported_claims: list[str],
    high_risk_keywords: list[str],
    conflicting_history: bool = False,
    repeated_failure: bool = False,
) -> EscalationDecision:
    """Apply escalation rules in priority order; the first match wins so the
    reported reason_code reflects the most specific/serious trigger."""

    if intent == "OTHER_OR_UNKNOWN":
        return EscalationDecision(
            "ESCALATE", rc.UNKNOWN_INTENT,
            "Classifier could not confidently map this message to a known intent.",
        )

    if intent_confidence < confidence_threshold:
        return EscalationDecision(
            "ESCALATE", rc.LOW_INTENT_CONFIDENCE,
            f"Intent confidence {intent_confidence:.2f} is below the "
            f"configured threshold {confidence_threshold:.2f}.",
        )

    hit = _has_high_risk_keyword(customer_message, SENSITIVE_ACCOUNT_KEYWORDS)
    if hit:
        return EscalationDecision(
            "ESCALATE", rc.SENSITIVE_ACCOUNT_ACTION,
            f"Message references a sensitive/irreversible account action ('{hit}').",
        )

    hit = _has_high_risk_keyword(customer_message, high_risk_keywords)
    if hit:
        return EscalationDecision(
            "ESCALATE", rc.HIGH_RISK_REQUEST,
            f"Message contains a high-risk keyword ('{hit}') requiring human review.",
        )

    if repeated_failure:
        return EscalationDecision(
            "ESCALATE", rc.REPEATED_FAILURE,
            "Customer appears to have had a prior unresolved interaction on this issue.",
        )

    if conflicting_history:
        return EscalationDecision(
            "ESCALATE", rc.CONFLICTING_HISTORY,
            "Retrieved historical examples give conflicting guidance for this situation.",
        )

    if retrieved_top_score < min_hybrid_score_for_grounding:
        return EscalationDecision(
            "ESCALATE", rc.NO_RELEVANT_HISTORY,
            "No sufficiently similar resolved historical interaction was found.",
        )

    if not generation_grounded or generation_unsupported_claims:
        return EscalationDecision(
            "ESCALATE", rc.UNSUPPORTED_INFORMATION,
            "Draft reply was not fully grounded in retrieved historical evidence.",
        )

    return EscalationDecision(
        "AUTO_HANDLE", None,
        "Intent confidence and retrieval evidence both meet thresholds, and the "
        "drafted reply is grounded in historical resolutions with no high-risk signals.",
    )
