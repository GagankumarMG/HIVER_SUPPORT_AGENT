# Golden Set Annotation Guide

This guide is for the human annotator labeling `data/golden/golden_candidates.csv`
via `streamlit run app/streamlit_app.py` (Annotate mode). Output is saved to
`data/golden/golden_set.csv`.

## What you're labeling

For each `customer_text`, you will assign:

1. **gold_intent** — one of the intent names in `configs/intents.yaml`
   (after intent discovery has been run and clusters named; use `OTHER_OR_UNKNOWN`
   if genuinely ambiguous or brand-irrelevant).
2. **gold_escalate** — `AUTO_HANDLE` or `ESCALATE`.
3. **gold_escalation_reason** — one of the reason codes in
   `src/escalation/reason_codes.py` (required only if `gold_escalate == ESCALATE`).
4. **gold_reply_quality** — free-text 1-5 rating of the BEST plausible reply an
   agent could give using only the retrieved historical context shown to you
   (not a rating of any specific model's output — this is a difficulty/quality
   ceiling label used later for calibration, distinct from the LLM-judge scores).
5. **annotator_notes** — anything ambiguous, worth flagging for future review.

## Intent definitions

See `configs/intents.yaml` for the authoritative, data-derived definitions,
inclusion rules, and exclusion rules for each intent. Read the whole file
before starting; when two intents seem to both apply, follow the exclusion
rules there to pick exactly one.

## Escalation criteria — label `ESCALATE` when:

- The message requires account-specific action (refund approval, cancellation,
  irreversible account change) that a generic reply cannot safely resolve.
- No historical example shown provides genuinely applicable guidance.
- The message reflects a customer who appears to have already tried the
  "obvious" fix (repeated/escalated frustration).
- The message involves legal, safety, fraud, or financial-dispute language.
- You are not confident a good-faith, policy-safe, non-fabricated reply is
  possible from the shown context alone.

Otherwise label `AUTO_HANDLE`.

## What counts as a "grounded" response

A response is grounded if every specific claim in it (a price, a timeframe, a
policy, an account action, a promise) is either:
- directly present in the customer's own message, or
- directly present in one of the retrieved historical support replies shown
  alongside the candidate.

## What counts as an unsupported claim

Any specific commitment (refund amount, exact resolution time, promise of a
callback, statement about "our policy") that does NOT appear in the customer
message or retrieved historical examples. When rating `gold_reply_quality`,
penalize an otherwise fluent reply that contains unsupported specifics.

## Handling ambiguous cases

- If two intents seem equally plausible, pick the one whose `inclusion_rules`
  more specifically match the wording used, and note the ambiguity in
  `annotator_notes`.
- If the message is too short/garbled to interpret confidently
  (e.g. "thanks!", "lol ok"), label intent `OTHER_OR_UNKNOWN` and escalate
  decision `AUTO_HANDLE` only if a generic acknowledgement is clearly safe;
  otherwise `ESCALATE` with reason `UNKNOWN_INTENT`.
- When in doubt, prefer `ESCALATE` — the assignment explicitly optimizes for
  safe automation over aggressive automation.

## Process

1. Launch `streamlit run app/streamlit_app.py`, choose "Annotate golden set."
2. For each example: read `customer_text` + `context`, pick intent, pick
   escalate/auto-handle + reason if escalating, rate reply-quality ceiling,
   add notes if needed, click Save & Next.
3. When all examples are labeled, run `python scripts/check_golden_set.py`
   to validate completeness before running evaluation.

**All 200 examples must be labeled by a human. Do not use model predictions
as a substitute — this would violate the assignment's evidence requirements
and invalidate every downstream metric.**
