#!/usr/bin/env python
"""Run the LLM-as-judge over the main system's golden-set replies (produced
by scripts/08_evaluate.py -> outputs/examples/main_system_eval_rows.csv).

If OPENAI_API_KEY is not set, writes a PENDING status instead of scores.

Usage:
    python scripts/09_run_llm_judge.py
"""
from __future__ import annotations

import json

import pandas as pd

from src.evaluation.llm_judge import judge_reply
from src.generation.llm_client import LLMClient
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    settings = load_settings()
    llm_client = LLMClient(settings.openai_api_key, settings.openai_model)

    eval_rows_path = "outputs/examples/main_system_eval_rows.csv"
    try:
        eval_rows = pd.read_csv(eval_rows_path)
    except FileNotFoundError:
        print(f"{eval_rows_path} not found. Run scripts/08_evaluate.py first.")
        return

    if not llm_client.available:
        status = {
            "status": "PENDING_NO_OPENAI_API_KEY",
            "note": "Set OPENAI_API_KEY in .env and re-run this script to compute "
                    "real LLM-judge scores. No scores were fabricated.",
        }
        with open("outputs/metrics/llm_judge_scores.json", "w") as f:
            json.dump(status, f, indent=2)
        print(json.dumps(status, indent=2))
        return

    scored = []
    for _, row in eval_rows.iterrows():
        result = judge_reply(
            llm_client,
            customer_message=row["customer_text"],
            candidate_reply=row["draft_reply"],
            historical_evidence=[row.get("retrieved_top_example", "")],
        )
        record = {"example_id": row["example_id"], "status": result.status}
        if result.scores:
            record.update(result.scores)
        scored.append(record)

    scored_df = pd.DataFrame(scored)
    scored_df.to_csv("outputs/examples/llm_judge_scores.csv", index=False)

    ok = scored_df[scored_df["status"] == "ok"]
    summary = {
        "status": "ok" if not ok.empty else "all_failed",
        "n_scored": int(len(ok)),
        "n_total": int(len(scored_df)),
    }
    for dim in ["groundedness", "relevance", "correctness", "helpfulness", "tone", "overall"]:
        if dim in ok.columns and not ok.empty:
            summary[f"mean_{dim}"] = float(ok[dim].astype(float).mean())

    with open("outputs/metrics/llm_judge_scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
