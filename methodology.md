# Methodology

## Pipeline overview

```
Raw Kaggle CSV (twcs.csv)
  -> schema validation & inspection            (scripts/00_inspect_dataset.py)
  -> load + parse                              (scripts/01_prepare_data.py)
  -> brand selection + thread reconstruction   (scripts/02_select_brand.py)
  -> intent discovery (clustering)             (scripts/03_discover_intents.py)
       -> HUMAN reviews clusters, names intents in configs/intents.yaml
  -> golden candidate sampling                 (scripts/04_create_golden_candidates.py)
       -> HUMAN labels via Streamlit annotation UI -> data/golden/golden_set.csv
       -> scripts/check_golden_set.py validates completeness
  -> chronological split + classifier training (scripts/05_train.py)
  -> retrieval index build (TRAIN split only)  (scripts/06_build_index.py)
  -> evaluation: Baseline A, B, Main System    (scripts/08_evaluate.py)
  -> LLM judge (optional, needs API key)       (scripts/09_run_llm_judge.py)
  -> human-agreement check (needs human labels)(scripts/09b_human_agreement.py)
  -> failure-mode analysis                     (scripts/10_analyze_failures.py)
  -> leakage checks                            (scripts/12_leakage_check.py)
  -> report assembly                           (scripts/11_generate_report.py)
  -> final quality gate                        (scripts/13_quality_gate.py)
```

## Brand selection

See `src/data/select_brand.py`. Candidates are authors whose messages are
≥95% outbound and who replied to ≥50 distinct customers with a resolvable
`in_response_to_tweet_id`. Ranked by
`n_complete_threads * 1.0 + n_distinct_customers * 1.5`.

## Thread reconstruction

`src/data/reconstruct_threads.py` joins each brand outbound tweet to the
customer tweet it replied to via `in_response_to_tweet_id`, handling:
missing/deleted target tweets, repeated tweet_ids (dedup, keep first),
multi-id malformed fields (comma-joined — first token used), and rows whose
target ID resolves to another outbound (non-customer) tweet (dropped).

## Preprocessing

`src/data/preprocess.py` is intentionally conservative: strips URLs and the
addressed @mention, masks emails/phone numbers, collapses whitespace. No
stopword removal, stemming, or lemmatization — these were judged more likely
to destroy intent-bearing signal than to help a linear classifier or
embedding model, both of which handle raw natural language well.

## Intent taxonomy

Discovered via KMeans clustering (6–10 clusters) over
SentenceTransformer embeddings (falls back to TF-IDF if unavailable) of
cleaned customer messages. `scripts/03_discover_intents.py` prints
representative (closest-to-centroid) examples per cluster; a human names
each cluster in `configs/intents.yaml`, including inclusion/exclusion rules.
An explicit `OTHER_OR_UNKNOWN` catch-all is always included.

## Classification

Two systems are compared:
- **Baseline B**: TF-IDF (1-2 grams) + Logistic Regression
  (`src/intents/classifier.py::TfidfLogisticClassifier`)
- **Main system**: SentenceTransformer embeddings (MiniLM-L6-v2) +
  calibrated Logistic Regression (`CalibratedClassifierCV`, sigmoid)
  (`src/intents/classifier.py::SemanticClassifier`)

Both fall back gracefully (TF-IDF embeddings) if `sentence-transformers`
cannot load a model (e.g. no internet at inference time), so the pipeline
degrades rather than crashing.

A configurable confidence threshold (`configs/config.yaml`
`classifier.confidence_threshold`, default 0.60) maps low-confidence
predictions to `OTHER_OR_UNKNOWN`.

## Retrieval

`src/retrieval/hybrid_retriever.py` combines:
- **Lexical**: BM25-Okapi (`rank_bm25`, with a pure-Python fallback
  implementation if the package is unavailable) over tokenized customer
  messages.
- **Semantic**: cosine similarity over SentenceTransformer embeddings
  (TF-IDF fallback).

Combined as `0.4 * lexical + 0.6 * semantic` (configurable). The index is
built exclusively from the TRAIN split (`scripts/06_build_index.py`);
`scripts/12_leakage_check.py` verifies no golden-set text ever appears in
the built index.

## Generation

`src/generation/reply_generator.py`:
- If `OPENAI_API_KEY` is set, calls the configured model with a strict
  grounding system prompt (`src/generation/prompts.py`) and structured JSON
  output requesting `reply`, `grounded`, `confidence`, `unsupported_claims`,
  `recommended_action`, `escalation_reason`.
- Otherwise (or on any LLM failure), falls back to returning the best
  retrieved historical reply as-is if its hybrid retrieval score clears
  `min_hybrid_score_for_grounding` (default 0.35); otherwise returns a fixed
  safe escalation message.

## Escalation

`src/escalation/policy.py` applies a prioritized rule list (unknown intent →
low confidence → sensitive account action → high-risk keyword → repeated
failure → conflicting history → no relevant history → ungrounded generation
→ else AUTO_HANDLE), each tagged with one of eight reason codes
(`src/escalation/reason_codes.py`).

## Evaluation

- **Classification**: accuracy, macro/weighted F1, per-intent P/R/F1
  (`src/intents/evaluate.py`).
- **Escalation**: treated as binary classification; reports
  `unsafe_auto_handling_rate` (gold=ESCALATE, predicted=AUTO_HANDLE) as the
  key safety metric, alongside automation/escalation rates
  (`src/evaluation/escalation_metrics.py`).
- **Reply quality**: LLM-as-judge on 6 dimensions
  (`src/evaluation/llm_judge.py`), validated against human scores on a
  50-example subset (`src/evaluation/human_agreement.py`), reporting exact
  agreement, mean absolute difference, Pearson correlation, and weighted
  Cohen's kappa per dimension.
- **Failure analysis**: real held-out rows only, categorized by rule-based
  heuristics (`src/evaluation/error_analysis.py`), never invented examples.

## Reproducibility

`scripts/14_run_metadata.py` records Python/package versions, seed, brand,
threshold, and retrieval weights to `outputs/metrics/run_metadata.json`.
All randomness is seeded (`configs/config.yaml` `seed: 42`) via
`src/utils/logging_utils.py::set_global_seed`.
