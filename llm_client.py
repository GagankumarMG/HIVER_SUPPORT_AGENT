"""Thin wrapper around the OpenAI SDK. Optional: if OPENAI_API_KEY is unset,
callers should use the fallback generator instead (see reply_generator.py).
"""
from __future__ import annotations

import json
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMUnavailableError("OPENAI_API_KEY not set.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Call the chat completion endpoint and parse a JSON response, with
        safe fallback parsing if the model wraps JSON in markdown fences or
        adds stray text."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return safe_parse_json(raw)


def safe_parse_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON parsing: strips markdown fences, extracts the first
    {...} block if there's stray text around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse LLM JSON output; returning parse-failure fallback.")
    return {
        "reply": None,
        "grounded": False,
        "confidence": 0.0,
        "unsupported_claims": ["LLM_JSON_PARSE_FAILURE"],
        "recommended_action": "ESCALATE",
        "escalation_reason": "LLM output could not be parsed as valid JSON.",
    }
