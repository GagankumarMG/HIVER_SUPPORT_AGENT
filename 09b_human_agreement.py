#!/usr/bin/env python
"""Compute agreement between the LLM judge and human scores on the
human-evaluation subset. Requires:
  - outputs/examples/llm_judge_scores.csv (from scripts/09_run_llm_judge.py)
  - data/golden/human_judge_scores.csv (HUMAN-COMPLETED; see below)

data/golden/human_judge_scores.csv must be created by a human rater scoring
a sample (default 50, configs/config.yaml evaluation.human_agreement_sample_size)
of the SAME examples the LLM judge scored, using the identical rubric in
src/evaluation/llm_judge.py::JUDGE_SYSTEM_PROMPT. Columns required:
  example_id, groundedness, relevance, correctness, helpfulness, tone, overall

This script NEVER fabricates human scores. If the file is missing or
incomplete, it writes a PENDING status with exact instructions.

Usage:
    python scripts/09b_human_agreement.py
"""
from __future__ import annotations

import json

import pandas as pd

from src.evaluation.human_agreement import compute_agreement, load_human_scores
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    settings = load_settings()
    sample_size = settings.get("evaluation", "human_agreement_sample_size", default=50)

    human_df = load_human_scores("data/golden/human_judge_scores.csv")
    llm_scores_path = "outputs/examples/llm_judge_scores.csv"

    try:
        llm_df = pd.read_csv(llm_scores_path)
        llm_df = llm_df[llm_df["status"] == "ok"]
    except FileNotFoundError:
        llm_df = None

    if human_df is None or llm_df is None or llm_df.empty:
        status = {
            "status": "PENDING_HUMAN_ANNOTATION",
            "instructions": [
                f"1. Run scripts/09_run_llm_judge.py with OPENAI_API_KEY set to "
                f"produce outputs/examples/llm_judge_scores.csv.",
                f"2. Sample {sample_size} example_ids from that file.",
                "3. Have a human rate the SAME examples on the SAME rubric "
                "(src/evaluation/llm_judge.py JUDGE_SYSTEM_PROMPT) and save to "
                "data/golden/human_judge_scores.csv with columns: "
                "example_id, groundedness, relevance, correctness, helpfulness, tone, overall",
                "4. Re-run this script.",
            ],
        }
        with open("outputs/metrics/judge_human_agreement.json", "w") as f:
            json.dump(status, f, indent=2)
        print(json.dumps(status, indent=2))
        return

    agreement = compute_agreement(human_df, llm_df)
    with open("outputs/metrics/judge_human_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)
    print(json.dumps(agreement, indent=2))

    _plot_agreement(human_df, llm_df)


def _plot_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = human_df.merge(llm_df, on="example_id", suffixes=("_human", "_llm"))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged["overall_human"], merged["overall_llm"], alpha=0.6)
    ax.plot([1, 5], [1, 5], "r--", label="perfect agreement")
    ax.set_xlabel("Human overall score")
    ax.set_ylabel("LLM judge overall score")
    ax.set_title("LLM Judge vs Human: Overall Score Agreement")
    ax.legend()
    plt.tight_layout()
    fig.savefig("outputs/figures/judge_vs_human.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
