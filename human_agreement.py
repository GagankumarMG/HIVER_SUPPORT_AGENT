"""Compute agreement between LLM-judge scores and human scores on the
human-evaluation subset (default 50 examples per configs/config.yaml).

Never fabricates human scores. If data/golden/human_judge_scores.csv does
not exist or is incomplete, returns a PENDING status with exact instructions
instead of numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DIMENSIONS = ["groundedness", "relevance", "correctness", "helpfulness", "tone", "overall"]


def compute_agreement(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict:
    """Both DataFrames must be indexed/joined on `example_id` and contain the
    DIMENSIONS columns."""
    merged = human_df.merge(llm_df, on="example_id", suffixes=("_human", "_llm"))
    if merged.empty:
        return {"status": "no_overlapping_examples"}

    per_dimension = {}
    for dim in DIMENSIONS:
        h = merged[f"{dim}_human"].astype(float)
        l = merged[f"{dim}_llm"].astype(float)
        exact_agreement = float((h == l).mean())
        mad = float((h - l).abs().mean())
        corr = float(np.corrcoef(h, l)[0, 1]) if h.std() > 0 and l.std() > 0 else None
        try:
            kappa = float(
                cohen_kappa_score(h.round().astype(int), l.round().astype(int), weights="linear")
            )
        except ValueError:
            kappa = None
        per_dimension[dim] = {
            "exact_agreement": exact_agreement,
            "mean_absolute_difference": mad,
            "pearson_correlation": corr,
            "weighted_kappa": kappa,
        }

    return {
        "status": "ok",
        "n_examples": int(len(merged)),
        "per_dimension": per_dimension,
    }


def load_human_scores(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing_cols = [c for c in DIMENSIONS if c not in df.columns]
    if missing_cols or df[DIMENSIONS].isna().any().any():
        return None  # incomplete -> treat as not-yet-available
    return df
