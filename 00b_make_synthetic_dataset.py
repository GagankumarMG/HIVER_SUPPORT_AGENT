#!/usr/bin/env python
"""Generate a SYNTHETIC demo dataset shaped like the Kaggle "Customer Support
on Twitter" export.

WHY THIS SCRIPT EXISTS
-----------------------
This repository's evaluation runner has no internet access to download the
real Kaggle dataset. This script fabricates a small, clearly-synthetic
dataset with the SAME SCHEMA so that:

  1. Every script in this repo (ingestion, brand selection, intent discovery,
     training, retrieval, evaluation, reporting) can be exercised end-to-end
     and actually produce real, reproducible numbers -- just on synthetic
     data instead of the real dataset.
  2. A reviewer can verify the whole pipeline works mechanically in minutes.

THIS IS NOT A SUBSTITUTE FOR THE REAL DATASET. All metrics produced from
this synthetic data are clearly labeled "SYNTHETIC_DEMO" in outputs/, and
must not be reported as real evaluation results. To get real results:

  1. Download the dataset from
     https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  2. Place the CSV at data/raw/twcs.csv (or set DATA_PATH in .env)
  3. Re-run scripts/00_inspect_dataset.py onward as normal.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger, set_global_seed

logger = get_logger(__name__)

BRAND_AUTHOR_ID = "SyntheticBrandSupport"
N_CUSTOMERS = 260
BASE_DATE = pd.Timestamp("2023-01-01", tz="UTC")

# Intent templates: (intent_name, customer_message_templates, support_reply_templates)
INTENT_TEMPLATES = {
    "PAYMENT_ISSUE": {
        "customer": [
            "My payment for order {oid} failed but the amount was deducted from my card.",
            "I was charged twice for order {oid}, can you check my payment?",
            "Payment of ${amt} did not go through but my bank shows it was taken.",
            "Why was I billed ${amt} when my plan should cost less?",
        ],
        "support": [
            "Sorry about that! We can see the duplicate charge for order {oid} and have "
            "flagged it for our billing team to reverse within 3-5 business days.",
            "We understand the frustration. Please DM us your order {oid} and email on "
            "file so we can verify the payment and correct the charge.",
        ],
    },
    "REFUND_REQUEST": {
        "customer": [
            "I want a refund for order {oid}, it arrived damaged.",
            "Please refund my subscription, I was charged after I cancelled.",
            "Can I get my money back for order {oid}? It's not what I expected.",
        ],
        "support": [
            "We're sorry to hear that! We've started a refund for order {oid}; it should "
            "reflect in your account within 5-7 business days.",
            "Thanks for flagging this - we'll process your refund for order {oid} right away.",
        ],
    },
    "LOGIN_OR_ACCOUNT": {
        "customer": [
            "I can't log into my account, it keeps saying wrong password.",
            "Locked out of my account after resetting my password, please help.",
            "Two factor code never arrives, I can't access my account.",
        ],
        "support": [
            "So sorry for the trouble logging in! Please try the password reset link at "
            "our help center; if that fails, DM your registered email and we'll assist.",
            "We can help with account access - please send your registered email via DM "
            "so we can look into the 2FA issue.",
        ],
    },
    "ORDER_OR_TRANSACTION": {
        "customer": [
            "What's the status of order {oid}? It's been a few days.",
            "Order {oid} shows processing for a week now, any update?",
        ],
        "support": [
            "Let us check on order {oid} for you - please DM your order confirmation "
            "email so we can look into the current status.",
            "Thanks for your patience! Order {oid} status can take up to 48 hours to "
            "update; we'll follow up if there's any delay.",
        ],
    },
    "DELIVERY_OR_SHIPPING": {
        "customer": [
            "My package for order {oid} was marked delivered but I never received it.",
            "Order {oid} arrived damaged in the box, what should I do?",
            "Tracking hasn't updated for order {oid} in 5 days.",
        ],
        "support": [
            "We're sorry to hear that! Please DM us order {oid} and we'll open an "
            "investigation with the carrier right away.",
            "That's not the experience we want for you - we'll arrange a replacement "
            "for order {oid} once you confirm your shipping address via DM.",
        ],
    },
    "TECHNICAL_PROBLEM": {
        "customer": [
            "The app keeps crashing when I try to check out.",
            "Getting error code 502 every time I open the dashboard.",
            "The search feature on the site is completely broken for me.",
        ],
        "support": [
            "Sorry for the disruption! Could you tell us your device/app version via DM "
            "so our tech team can investigate the crash?",
            "We're aware some users are seeing this error and our team is investigating - "
            "thanks for your patience.",
        ],
    },
    "CANCELLATION": {
        "customer": [
            "Please cancel my subscription, I no longer need it.",
            "How do I cancel order {oid} before it ships?",
        ],
        "support": [
            "We're sorry to see you go! We've submitted the cancellation request - you'll "
            "get a confirmation email shortly.",
            "We can help cancel order {oid} if it hasn't shipped yet - please DM your "
            "order email to confirm.",
        ],
    },
    "PRODUCT_INFORMATION": {
        "customer": [
            "Does your product work with international accounts?",
            "What's the difference between the basic and premium plans?",
            "Is there a student discount available?",
        ],
        "support": [
            "Great question! Yes, our product supports international accounts - happy to "
            "share more details if needed.",
            "Premium includes priority support and extra features over Basic - check our "
            "pricing page for the full comparison!",
        ],
    },
}

OTHER_MESSAGES = [
    "just saying hi, love your product!",
    "lol nice ad",
    "thanks for the follow back",
]


def _rand_choice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def generate(n_customers: int = N_CUSTOMERS, seed: int = 42) -> pd.DataFrame:
    set_global_seed(seed)
    rng = random.Random(seed)
    rows = []
    tweet_id_counter = 1

    intents = list(INTENT_TEMPLATES.keys())

    for i in range(n_customers):
        intent = _rand_choice(rng, intents) if rng.random() > 0.05 else None
        oid = f"{10000 + i}"
        amt = rng.choice([9.99, 19.99, 49.99, 99.0, 150.0])
        cust_author = f"customer_{i:04d}"
        day_offset = int(i * (300 / n_customers))  # spread over ~300 days
        ts = BASE_DATE + pd.Timedelta(days=day_offset, hours=rng.randint(0, 23))

        if intent is None:
            text = _rand_choice(rng, OTHER_MESSAGES)
        else:
            tmpl = _rand_choice(rng, INTENT_TEMPLATES[intent]["customer"])
            text = tmpl.format(oid=oid, amt=amt)

        cust_tweet_id = str(tweet_id_counter)
        tweet_id_counter += 1
        rows.append(
            {
                "tweet_id": cust_tweet_id,
                "author_id": cust_author,
                "inbound": True,
                "created_at": ts.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": f"@{BRAND_AUTHOR_ID} {text}",
                "response_tweet_id": None,
                "in_response_to_tweet_id": None,
            }
        )

        # ~90% of customer messages get a support response (some don't, to
        # exercise "missing response" handling in reconstruct_threads.py)
        if rng.random() < 0.90 and intent is not None:
            reply_tmpl = _rand_choice(rng, INTENT_TEMPLATES[intent]["support"])
            reply_text = reply_tmpl.format(oid=oid, amt=amt)
            support_tweet_id = str(tweet_id_counter)
            tweet_id_counter += 1
            reply_ts = ts + pd.Timedelta(minutes=rng.randint(5, 240))
            rows.append(
                {
                    "tweet_id": support_tweet_id,
                    "author_id": BRAND_AUTHOR_ID,
                    "inbound": False,
                    "created_at": reply_ts.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                    "text": f"@{cust_author} {reply_text}",
                    "response_tweet_id": None,
                    "in_response_to_tweet_id": cust_tweet_id,
                }
            )
            # backfill response_tweet_id on the customer row
            rows[-2]["response_tweet_id"] = support_tweet_id

        # occasional malformed row: comma-joined ids (data quality edge case)
        if rng.random() < 0.02 and len(rows) >= 2:
            rows[-1]["in_response_to_tweet_id"] = f"{cust_tweet_id},999999"

    df = pd.DataFrame(rows)
    return df


def main():
    out_path = Path("data/raw/twcs_synthetic.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out_path, index=False)
    logger.info("Wrote synthetic dataset: %d rows -> %s", len(df), out_path)
    print(f"SYNTHETIC dataset written to {out_path} ({len(df)} rows).")
    print("This is fabricated demo data, NOT the real Kaggle dataset.")
    print("Set DATA_PATH=data/raw/twcs_synthetic.csv in .env to use it for a fast demo run,")
    print("or replace with the real twcs.csv for actual evaluation.")


if __name__ == "__main__":
    main()
