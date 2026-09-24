"""Brand/company account selection heuristic.

Company/support accounts in the Kaggle dataset are the `author_id` values
that appear almost exclusively as OUTBOUND (inbound == False) authors and
reply frequently to many distinct customers. We rank candidates by a
composite of conversation volume and thread completeness so the selected
brand has enough data for training, retrieval, and a 200-example golden set.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MIN_OUTBOUND_SHARE = 0.95  # candidate must be outbound almost all the time
MIN_DISTINCT_CUSTOMERS = 50


def rank_brand_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of candidate brand accounts ranked by usable volume."""
    outbound = df[~df["inbound"]].copy()
    inbound = df[df["inbound"]].copy()

    author_stats = (
        outbound.groupby("author_id")
        .agg(n_outbound=("tweet_id", "count"))
        .reset_index()
    )

    # Share of this author's total messages that are outbound (company-like).
    total_by_author = df.groupby("author_id")["tweet_id"].count().rename("n_total")
    author_stats = author_stats.merge(total_by_author, on="author_id", how="left")
    author_stats["outbound_share"] = author_stats["n_outbound"] / author_stats["n_total"]

    # Count distinct customers each candidate replied to, using
    # in_response_to_tweet_id -> map back to the inbound author.
    inbound_by_tweet = inbound.set_index("tweet_id")["author_id"].to_dict()
    outbound_local = outbound.dropna(subset=["in_response_to_tweet_id"]).copy()
    outbound_local["responded_to_customer"] = outbound_local[
        "in_response_to_tweet_id"
    ].map(inbound_by_tweet)
    thread_stats = (
        outbound_local.dropna(subset=["responded_to_customer"])
        .groupby("author_id")
        .agg(
            n_complete_threads=("responded_to_customer", "count"),
            n_distinct_customers=("responded_to_customer", "nunique"),
        )
        .reset_index()
    )

    candidates = author_stats.merge(thread_stats, on="author_id", how="left").fillna(
        {"n_complete_threads": 0, "n_distinct_customers": 0}
    )
    candidates = candidates[candidates["outbound_share"] >= MIN_OUTBOUND_SHARE]
    candidates = candidates[candidates["n_distinct_customers"] >= MIN_DISTINCT_CUSTOMERS]

    # Composite score: weight conversation volume and thread completeness.
    candidates["score"] = (
        candidates["n_complete_threads"] * 1.0
        + candidates["n_distinct_customers"] * 1.5
    )
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    return candidates


def select_brand(
    df: pd.DataFrame, forced_handle: str | None = None
) -> tuple[str, pd.DataFrame]:
    candidates = rank_brand_candidates(df)
    if candidates.empty:
        raise ValueError(
            "No brand candidates met the minimum thresholds "
            f"(outbound_share>={MIN_OUTBOUND_SHARE}, "
            f"distinct_customers>={MIN_DISTINCT_CUSTOMERS}). "
            "Lower thresholds in src/data/select_brand.py or supply a larger dataset."
        )
    if forced_handle:
        if forced_handle not in candidates["author_id"].values:
            raise ValueError(
                f"BRAND_HANDLE={forced_handle} not found among viable candidates. "
                f"Top candidates: {candidates['author_id'].head(10).tolist()}"
            )
        selected = forced_handle
        logger.info("Using forced BRAND_HANDLE=%s", selected)
    else:
        selected = candidates.iloc[0]["author_id"]
        logger.info("Auto-selected brand author_id=%s", selected)
    return selected, candidates
