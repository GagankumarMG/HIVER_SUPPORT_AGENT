import pandas as pd
import pytest

from src.data.load_dataset import SchemaError, validate_schema
from src.data.preprocess import clean_text, mask_pii
from src.data.reconstruct_threads import reconstruct_pairs


def test_validate_schema_passes_with_expected_columns():
    df = pd.DataFrame(columns=[
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ])
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame(columns=["tweet_id", "text"])
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_clean_text_strips_urls_and_whitespace():
    raw = "Check this out  https://example.com/path   now"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "  " not in cleaned


def test_mask_pii_masks_email_and_phone():
    text = "email me at test@example.com or call 555-123-4567"
    masked = mask_pii(text)
    assert "test@example.com" not in masked
    assert "[EMAIL]" in masked


def test_reconstruct_pairs_handles_malformed_and_missing_ids():
    df = pd.DataFrame(
        [
            {"tweet_id": "1", "author_id": "cust1", "inbound": True,
             "text": "help", "response_tweet_id": "2",
             "in_response_to_tweet_id": None, "created_at_parsed": "2023-01-01"},
            {"tweet_id": "2", "author_id": "BRAND", "inbound": False,
             "text": "reply", "response_tweet_id": None,
             "in_response_to_tweet_id": "1", "created_at_parsed": "2023-01-01"},
            # malformed: comma-joined id, first token doesn't exist
            {"tweet_id": "3", "author_id": "BRAND", "inbound": False,
             "text": "orphan reply", "response_tweet_id": None,
             "in_response_to_tweet_id": "999,888", "created_at_parsed": "2023-01-01"},
        ]
    )
    pairs = reconstruct_pairs(df, "BRAND")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_text"] == "help"
    assert pairs.iloc[0]["support_text"] == "reply"
