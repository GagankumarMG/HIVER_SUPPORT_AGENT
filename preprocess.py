"""Conservative text preprocessing.

Preserves sentiment, intent-bearing terms, product names, and error
descriptions. Only removes URLs / excess whitespace / obvious system
artifacts. No stopword removal, no stemming/lemmatization (these can destroy
information needed for intent classification and grounding checks).
"""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\+?\d[\d\-\s()]{7,}\d\b")


def mask_pii(text: str) -> str:
    """Mask obvious personal identifiers (emails, phone numbers)."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def clean_text(raw_text: str, keep_mentions: bool = False) -> str:
    """Produce a cleaned version of a tweet for modeling.

    - strips URLs
    - optionally strips @mentions (kept by default off since brand mentions
      can matter for context, but mentions are stripped for the FIRST mention
      which is typically the addressed account, since it adds no semantic value)
    - collapses whitespace
    - masks PII
    """
    if raw_text is None:
        return ""
    text = str(raw_text)
    text = URL_RE.sub("", text)
    if not keep_mentions:
        text = MENTION_RE.sub("", text)
    text = mask_pii(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def add_text_fields(df, text_col: str = "customer_text"):
    """Add raw_text and cleaned_text columns to a DataFrame in place-safe way."""
    df = df.copy()
    df["raw_text"] = df[text_col]
    df["cleaned_text"] = df[text_col].apply(clean_text)
    return df
