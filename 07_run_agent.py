#!/usr/bin/env python
"""Run the full agent pipeline on a single customer message from the CLI.

Usage:
    python scripts/07_run_agent.py "My payment failed but I was charged"
"""
from __future__ import annotations

import json
import sys

from src.agent.pipeline import PipelineConfig, SupportAgentPipeline
from src.generation.llm_client import LLMClient
from src.intents.classifier import SemanticClassifier
from src.retrieval.build_index import load_index
from src.utils.config import load_settings


def build_pipeline() -> SupportAgentPipeline:
    settings = load_settings()
    classifier = SemanticClassifier.load("models/main_semantic_classifier.joblib")
    retriever = load_index("models/hybrid_retrieval_index.joblib")
    llm_client = LLMClient(settings.openai_api_key, settings.openai_model)
    config = PipelineConfig(
        brand=settings.brand_config.get("brand", {}).get("display_name") or "Brand",
        confidence_threshold=settings.get("classifier", "confidence_threshold", default=0.60),
        min_hybrid_score_for_grounding=settings.get(
            "retrieval", "min_hybrid_score_for_grounding", default=0.35
        ),
        top_k_retrieval=settings.get("retrieval", "top_k", default=5),
        high_risk_keywords=settings.get("escalation", "high_risk_keywords", default=[]),
        max_tokens=settings.get("generation", "max_tokens", default=400),
        temperature=settings.get("generation", "temperature", default=0.2),
    )
    return SupportAgentPipeline(classifier, retriever, config, llm_client)


def main():
    message = " ".join(sys.argv[1:]) or "My order hasn't arrived and tracking hasn't updated in 5 days."
    pipeline = build_pipeline()
    result = pipeline.run(message)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
