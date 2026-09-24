"""LLM-as-judge for reply quality.

Requires OPENAI_API_KEY. If unset, `judge_reply` returns a clearly marked
"PENDING_LLM_JUDGE" result rather than a fabricated score, so downstream
reports never silently contain made-up numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.generation.llm_client import LLMClient, safe_parse_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of customer support replies.
Score the CANDIDATE REPLY against the customer message and the historical
support evidence on these dimensions, each 1-5 (5 = best):

- groundedness: Is the response supported by the historical evidence (no invented
  facts, prices, timelines, or policies)?
- relevance: Does it address the actual customer problem?
- correctness: Does it avoid unsupported claims?
- helpfulness: Would this response meaningfully help the customer?
- tone: Is it professional and appropriate?
- overall: Your holistic 1-5 judgment.

Return ONLY JSON:
{"groundedness": int, "relevance": int, "correctness": int, "helpfulness": int,
 "tone": int, "overall": int, "reason": "string"}
"""


@dataclass
class JudgeResult:
    scores: dict | None
    status: str  # "ok", "pending_no_api_key", "error"
    raw_error: str | None = None


def judge_reply(
    llm_client: LLMClient,
    customer_message: str,
    candidate_reply: str,
    historical_evidence: list[str],
) -> JudgeResult:
    if not llm_client.available:
        return JudgeResult(scores=None, status="pending_no_api_key")

    evidence_block = "\n".join(f"- {e}" for e in historical_evidence) or "(none retrieved)"
    user_prompt = f"""Customer message: {customer_message}

Historical support evidence:
{evidence_block}

Candidate reply: {candidate_reply}
"""
    try:
        parsed = llm_client.complete_json(JUDGE_SYSTEM_PROMPT, user_prompt, max_tokens=300, temperature=0.0)
        required = {"groundedness", "relevance", "correctness", "helpfulness", "tone", "overall"}
        if not required.issubset(parsed.keys()):
            return JudgeResult(scores=None, status="error", raw_error=f"missing keys: {parsed}")
        return JudgeResult(scores=parsed, status="ok")
    except Exception as e:  # noqa: BLE001
        logger.error("LLM judge call failed: %s", e)
        return JudgeResult(scores=None, status="error", raw_error=str(e))
