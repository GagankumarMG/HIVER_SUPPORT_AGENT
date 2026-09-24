"""Pydantic schemas for agent I/O, used across the pipeline, tests, and the
Streamlit demo so output shape is always validated."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TopIntent(BaseModel):
    intent: str
    score: float


class RetrievedExampleOut(BaseModel):
    customer_message: str
    historical_reply: str
    score: float


class AgentResponse(BaseModel):
    customer_message: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    top_intents: list[TopIntent] = Field(default_factory=list)
    retrieved_examples: list[RetrievedExampleOut] = Field(default_factory=list)
    draft_reply: str
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    decision: str
    escalation_reason: Optional[str] = None
    reason_code: Optional[str] = None
    generation_mode: Optional[str] = None
