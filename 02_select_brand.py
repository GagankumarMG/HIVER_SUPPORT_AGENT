#!/usr/bin/env python
"""Select the single brand/support account to build the agent for, and
reconstruct customer<->support pairs for that brand.

Usage:
    python scripts/02_select_brand.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.load_dataset import load_raw
from src.data.preprocess import add_text_fields
from src.data.reconstruct_threads import reconstruct_pairs
from src.data.select_brand import select_brand
from src.utils.config import load_settings
from src.utils.logging_utils import get_logger
from src.utils.io import load_table, save_table

logger = get_logger(__name__)


def main():
    settings = load_settings()
    df = load_raw(settings.data_path)

    brand_id, candidates = select_brand(df, forced_handle=settings.brand_handle)
    candidates.to_csv("outputs/metrics/brand_candidates.csv", index=False)
    logger.info("Saved brand candidate ranking to outputs/metrics/brand_candidates.csv")

    row = candidates[candidates["author_id"] == brand_id].iloc[0]
    brand_cfg = {
        "brand": {
            "id": str(brand_id),
            "display_name": str(brand_id),
            "selection_reason": (
                "Chosen using support-volume and thread-completeness heuristic: "
                "ranked by (n_complete_threads * 1.0 + n_distinct_customers * 1.5) "
                "among accounts with outbound_share >= 0.95 and "
                ">=50 distinct customers replied to. See src/data/select_brand.py."
            ),
            "n_conversations": int(row["n_complete_threads"]),
            "n_inbound": int(row["n_distinct_customers"]),
            "n_outbound": int(row["n_outbound"]),
        }
    }
    with open("configs/brand.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(brand_cfg, f, sort_keys=False)
    print(f"Selected brand: {brand_id}")
    print(f"  complete threads: {row['n_complete_threads']}")
    print(f"  distinct customers: {row['n_distinct_customers']}")

    pairs = reconstruct_pairs(df, brand_id)
    pairs = add_text_fields(pairs, text_col="customer_text")
    out_path = Path("data/processed/support_pairs.parquet")
    save_table(pairs, out_path)
    logger.info("Saved %d support pairs to %s", len(pairs), out_path)


if __name__ == "__main__":
    main()
