#!/usr/bin/env python
"""Run Baseline A, Baseline B, and the Main System against the golden set,
compute classification/escalation metrics, and save everything under
outputs/metrics + outputs/figures. This is the core reproducible evaluation
harness (Section 2/3 requirement: works WITHOUT an LLM API key).

Usage:
    python scripts/08_evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.agent.pipeline import PipelineConfig, SupportAgentPipeline
from src.evaluation.baselines import BaselineA, BaselineB
from src.evaluation.escalation_metrics import compute_escalation_metrics
from src.evaluation.reply_metrics import historical_similarity
from src.intents.classifier import SemanticClassifier
from src.intents.evaluate import compute_classification_metrics, plot_confusion_matrix
from src.generation.llm_client import LLMClient
from src.retrieval.build_index import load_index
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def load_golden() -> pd.DataFrame | None:
    path = Path("data/golden/golden_set.csv")
    if not path.exists():
        logger.warning(
            "data/golden/golden_set.csv not found. Cannot compute headline "
            "golden-set metrics. Run scripts/04_create_golden_candidates.py "
            "then annotate via the Streamlit app first."
        )
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["gold_intent", "gold_escalate"])
    if df.empty:
        return None
    return df


def main():
    settings = load_settings()
    golden = load_golden()
    if golden is None:
        status = {"status": "PENDING_GOLDEN_SET_ANNOTATION"}
        with open("outputs/metrics/evaluation_status.json", "w") as f:
            json.dump(status, f, indent=2)
        print("EVALUATION SKIPPED: golden set not yet annotated. See "
              "outputs/metrics/evaluation_status.json and README section "
              "'Golden set annotation'.")
        return

    train = load_table("data/processed/train.parquet")

    # ---- Baseline A ----
    baseline_a = BaselineA().fit(train["intent"].tolist())
    a_pred_intent, a_pred_decision, a_reply = [], [], []
    for text in golden["customer_text"]:
        r = baseline_a.predict(text)
        a_pred_intent.append(r.intent)
        a_pred_decision.append(r.decision)
        a_reply.append(r.reply)

    # ---- Baseline B ----
    corpus = pd.DataFrame(
        {
            "customer_text": train["cleaned_text"],
            "support_reply": train["support_text"],
        }
    ).dropna()
    baseline_b = BaselineB(
        confidence_threshold=settings.get("classifier", "confidence_threshold", default=0.60)
    ).fit(train["cleaned_text"].tolist(), train["intent"].tolist(), corpus)
    b_pred_intent, b_pred_decision, b_reply = [], [], []
    for text in golden["customer_text"]:
        r = baseline_b.predict(text)
        b_pred_intent.append(r.intent)
        b_pred_decision.append(r.decision)
        b_reply.append(r.reply)

    # ---- Main system ----
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
    )
    pipeline = SupportAgentPipeline(classifier, retriever, config, llm_client)

    main_pred_intent, main_pred_decision, main_reply, main_grounding_sim = [], [], [], []
    eval_rows = []
    for _, row in golden.iterrows():
        res = pipeline.run(row["customer_text"])
        main_pred_intent.append(res.intent)
        main_pred_decision.append(res.decision)
        main_reply.append(res.draft_reply)
        best_hist = res.retrieved_examples[0].historical_reply if res.retrieved_examples else ""
        main_grounding_sim.append(historical_similarity(res.draft_reply, best_hist))
        eval_rows.append(
            {
                "example_id": row["example_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],
                "pred_intent": res.intent,
                "gold_escalate": row["gold_escalate"],
                "pred_decision": res.decision,
                "draft_reply": res.draft_reply,
                "retrieved_top_example": best_hist,
                "reason_code": res.reason_code,
                "generation_mode": res.generation_mode,
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv("outputs/examples/main_system_eval_rows.csv", index=False)

    gold_intent = golden["gold_intent"].tolist()
    gold_escalate = golden["gold_escalate"].tolist()

    results = {}
    for name, pred_intent, pred_decision in [
        ("baseline_a", a_pred_intent, a_pred_decision),
        ("baseline_b", b_pred_intent, b_pred_decision),
        ("main_system", main_pred_intent, main_pred_decision),
    ]:
        cls_metrics = compute_classification_metrics(gold_intent, pred_intent)
        esc_metrics = compute_escalation_metrics(gold_escalate, pred_decision)
        payload = {"classification": cls_metrics, "escalation": esc_metrics}
        if name == "main_system":
            payload["reply_grounding_similarity_mean"] = (
                sum(main_grounding_sim) / len(main_grounding_sim) if main_grounding_sim else None
            )
            payload["judge_reply_quality"] = {
                "status": "PENDING_LLM_JUDGE",
                "note": "Run scripts/09_run_llm_judge.py with OPENAI_API_KEY set "
                        "to populate mean_groundedness/mean_helpfulness etc.",
            }
        with open(f"outputs/metrics/{name}_metrics.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        results[name] = payload
        plot_confusion_matrix(gold_intent, pred_intent, f"outputs/figures/{name}_confusion_matrix.png")
        logger.info("%s: macro_f1=%.3f accuracy=%.3f automation_rate=%s",
                     name, cls_metrics["macro_f1"], cls_metrics["accuracy"],
                     esc_metrics["automation_rate"])

    print(json.dumps(
        {k: {"macro_f1": v["classification"]["macro_f1"],
             "accuracy": v["classification"]["accuracy"],
             "automation_rate": v["escalation"]["automation_rate"],
             "unsafe_auto_handling_rate": v["escalation"]["unsafe_auto_handling_rate"]}
         for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
