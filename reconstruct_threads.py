"""Reconstruct customer -> support response pairs from raw tweet rows.

Handles: missing response ids, deleted/missing tweets, multiple responses
per customer tweet, repeated tweets, and malformed comma-separated id lists
(the raw dataset sometimes stores multiple candidate response ids joined by
commas in `response_tweet_id`).
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _first_valid_id(raw: str | float) -> str | None:
    """Parse a possibly comma-separated id field, returning the first token."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    # malformed rows sometimes contain multiple ids like "123,456"
    first = s.split(",")[0].strip()
    return first or None


def reconstruct_pairs(df: pd.DataFrame, brand_author_id: str) -> pd.DataFrame:
    """Build customer_text -> support_text pairs for a single brand.

    Returns a DataFrame with columns:
      conversation_id, customer_tweet_id, customer_text, support_tweet_id,
      support_text, timestamp, customer_author, support_author
    """
    df = df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)

    by_id = df.set_index("tweet_id", drop=False)
    by_id = by_id[~by_id.index.duplicated(keep="first")]  # repeated tweets guard

    support_rows = df[(df["author_id"] == brand_author_id) & (~df["inbound"])].copy()
    support_rows["in_response_to_tweet_id"] = support_rows[
        "in_response_to_tweet_id"
    ].apply(_first_valid_id)

    records = []
    n_missing_customer = 0
    n_malformed = 0

    for _, srow in support_rows.iterrows():
        cust_id = srow["in_response_to_tweet_id"]
        if cust_id is None:
            n_malformed += 1
            continue
        if cust_id not in by_id.index:
            n_missing_customer += 1
            continue
        crow = by_id.loc[cust_id]
        if isinstance(crow, pd.DataFrame):  # safety in case of residual dupes
            crow = crow.iloc[0]
        if not bool(crow["inbound"]):
            # the "customer" tweet id actually belongs to another outbound
            # message (e.g. support-to-support); skip, not a real customer pair.
            continue
        records.append(
            {
                "conversation_id": f"{crow['tweet_id']}__{srow['tweet_id']}",
                "customer_tweet_id": crow["tweet_id"],
                "customer_text": crow["text"],
                "support_tweet_id": srow["tweet_id"],
                "support_text": srow["text"],
                "timestamp": srow.get("created_at_parsed", srow.get("created_at")),
                "customer_author": crow["author_id"],
                "support_author": srow["author_id"],
            }
        )

    pairs = pd.DataFrame.from_records(records)
    logger.info(
        "Reconstructed %d customer-support pairs (missing_customer_tweet=%d, "
        "malformed_response_ids=%d) for brand=%s",
        len(pairs),
        n_missing_customer,
        n_malformed,
        brand_author_id,
    )
    return pairs
