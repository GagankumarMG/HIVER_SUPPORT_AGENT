#!/usr/bin/env python
"""Stratified-sample golden evaluation candidates for human annotation.

Samples across: intent (from weak/cluster labels), message length buckets,
and time period, targeting configs/config.yaml evaluation.golden_set_size
(default 200, within the allowed 150-250 range). Leaves all *gold_* label
columns EMPTY -- they must be filled in via the Streamlit annotation UI
(app/streamlit_app.py, "Annotate" mode) or manually.

Usage:
    python scripts/04_create_golden_candidates.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_settings
from src.utils.logging_utils import get_logger, set_global_seed
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def length_bucket(text: str) -> str:
    n = len(str(text).split())
    if n <= 4:
        return "short"
    if n <= 15:
        return "medium"
    return "long"


def main():
    settings = load_settings()
    set_global_seed(settings.seed)
    target_n = settings.get("evaluation", "golden_set_size", default=200)

    pairs = load_table("data/processed/support_pairs.parquet")
    pairs = pairs.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    pairs["length_bucket"] = pairs["cleaned_text"].apply(length_bucket)
    pairs["time_period"] = pd.qcut(
        pd.to_datetime(pairs["timestamp"], errors="coerce", utc=True).rank(method="first"),
        q=4, labels=["p1", "p2", "p3", "p4"],
    )

    strata_cols = ["length_bucket", "time_period"]
    n_strata = pairs.groupby(strata_cols, observed=True).ngroups
    per_stratum = max(1, target_n // max(1, n_strata))

    sampled = (
        pairs.groupby(strata_cols, observed=True, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=settings.seed))
    )
    if len(sampled) < target_n:
        remaining = pairs.drop(sampled.index)
        top_up = remaining.sample(
            min(len(remaining), target_n - len(sampled)), random_state=settings.seed
        )
        sampled = pd.concat([sampled, top_up])
    sampled = sampled.sample(min(len(sampled), target_n), random_state=settings.seed).reset_index(drop=True)

    candidates = pd.DataFrame(
        {
            "example_id": [f"golden_{i:04d}" for i in range(len(sampled))],
            "customer_text": sampled["cleaned_text"],
            "context": sampled["raw_text"],
            "candidate_intent": "",  # to be filled by classifier pre-labeling if desired
            "gold_intent": "",
            "gold_reply_quality": "",
            "gold_escalate": "",
            "gold_escalation_reason": "",
            "annotator_notes": "",
        }
    )
    candidates.to_csv("data/golden/golden_candidates.csv", index=False)
    logger.info("Wrote %d golden candidates to data/golden/golden_candidates.csv", len(candidates))
    print(f"Wrote {len(candidates)} golden candidates. Target was {target_n} "
          "(allowed range 150-250 per assignment spec).")
    print("NEXT STEP (human action required): run `streamlit run app/streamlit_app.py`, "
          "select 'Annotate golden set' mode, and label every example. "
          "This produces data/golden/golden_set.csv used by scripts/08_evaluate.py.")


if __name__ == "__main__":
    main()
