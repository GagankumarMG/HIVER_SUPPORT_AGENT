#!/usr/bin/env python
"""Run exploratory clustering over the brand's historical customer messages
to discover a data-grounded intent taxonomy. Prints representative examples
per cluster for a human to name; does NOT auto-write final intent names into
configs/intents.yaml (naming clusters is a deliberate human decision - see
docs/decision_log.md).

Usage:
    python scripts/03_discover_intents.py
"""
from __future__ import annotations

import json

import pandas as pd

from src.intents.discover import discover_intents
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def main():
    settings = load_settings()
    pairs = load_table("data/processed/support_pairs.parquet")
    texts = pairs["cleaned_text"].fillna("").tolist()

    n_clusters = settings.get("intents", "n_clusters_discovery", default=8)
    result = discover_intents(texts, n_clusters=n_clusters, seed=settings.seed)

    out = {
        "method": result.method,
        "silhouette_score": result.silhouette,
        "n_clusters": len(result.representative_examples),
        "representative_examples_per_cluster": result.representative_examples,
    }
    with open("outputs/metrics/intent_discovery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 60)
    print(f"Intent discovery complete: {out['n_clusters']} clusters "
          f"(method={out['method']}, silhouette={out['silhouette_score']:.3f})")
    print("Representative examples per cluster (review and name these in configs/intents.yaml):")
    for cluster_id, examples in result.representative_examples.items():
        print(f"\n--- Cluster {cluster_id} ---")
        for ex in examples:
            print(f"  - {ex}")
    print("=" * 60)
    print("Next step: update configs/intents.yaml with human-assigned names for "
          "each cluster, then use src/intents/label.py to apply the mapping "
          "before training (scripts/05_train.py).")


if __name__ == "__main__":
    main()
