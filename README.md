# HIVER_SUPPORT_AGENT

AI customer-support agent built from real Twitter support conversations,
combining intent classification, historical-response retrieval, grounded
reply drafting, and escalation decisions — with reproducible evaluation.

> **Read this first:** this repository was built and verified in an
> environment with **no internet access**, so the real Kaggle dataset could
> not be downloaded during development. A schema-matched **synthetic demo
> dataset** is bundled so the entire pipeline is provably runnable end-to-end
> in minutes (`scripts/00b_make_synthetic_dataset.py`). Every metric produced
> from that synthetic data is labeled accordingly; nothing is presented as a
> real evaluation result. Point `DATA_PATH` at the real CSV to get real
> results — no code changes needed. See "Limitations" below.

---

## 1. Problem

Given a brand's historical Twitter customer-support conversations, build an
agent that: classifies each incoming message's intent, drafts a reply
grounded in how that brand has actually resolved similar issues before, and
decides AUTO-HANDLE vs ESCALATE with a stated reason — with evidence, not
just architecture, that it works.

## 2. Architecture

```
Customer message
   -> preprocessing (URL/PII strip, whitespace normalize)
   -> intent classifier (SentenceTransformer + calibrated LogisticRegression,
      TF-IDF fallback; confidence threshold -> OTHER_OR_UNKNOWN)
   -> hybrid retrieval (0.4 BM25 + 0.6 semantic cosine, top-5, TRAIN-split only)
   -> grounded reply generation (LLM if OPENAI_API_KEY set, else best-retrieved
      historical reply, else safe generic escalation message)
   -> escalation policy (8 prioritized reason codes; see src/escalation/policy.py)
   -> AUTO_HANDLE or ESCALATE + reason
```

Full diagram and per-stage detail: `docs/methodology.md`.

## 3. Dataset setup

Download from
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
and place the CSV anywhere, then set `DATA_PATH` in `.env` (copy from
`.env.example`) to point at it. Expected columns: `tweet_id`, `author_id`,
`inbound`, `created_at`, `text`, `response_tweet_id`,
`in_response_to_tweet_id`. `src/data/load_dataset.py` validates this schema
and raises a clear error if columns are missing.

**No dataset yet / just want to see the pipeline run?** Use the bundled
synthetic demo dataset instead (see "Fast demo mode" below) — no download
needed.

## 4. Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Requires Python 3.11+.

## 5. Configuration

Edit `.env`:

```
DATA_PATH=data/raw/twcs.csv     # or data/raw/twcs_synthetic.csv for the demo
BRAND_HANDLE=                   # leave empty to auto-select
OPENAI_API_KEY=                 # optional; core pipeline works without it
OPENAI_MODEL=gpt-4o-mini
```

Tunable parameters (classifier threshold, retrieval weights, escalation
keywords, golden-set size, etc.) live in `configs/config.yaml`,
`configs/intents.yaml`, `configs/brand.yaml` — all documented inline.

## 6. Brand selection

```bash
python scripts/02_select_brand.py
```

Ranks candidate brand/support accounts by conversation volume and thread
completeness (`src/data/select_brand.py`), writes the ranking to
`outputs/metrics/brand_candidates.csv`, and saves the selected brand to
`configs/brand.yaml`. Also reconstructs customer↔support pairs for that
brand into `data/processed/support_pairs.parquet`.

## 7. Data preparation

```bash
python scripts/00_inspect_dataset.py     # schema + stats -> outputs/metrics/dataset_summary.json
python scripts/01_prepare_data.py        # load + parse -> data/interim/
python scripts/02_select_brand.py        # brand + thread reconstruction (see above)
python scripts/03_discover_intents.py    # clustering -> review output, name intents in configs/intents.yaml
```

## 8. Golden set annotation (REQUIRED human step)

```bash
python scripts/04_create_golden_candidates.py   # samples 150-250 candidates
streamlit run app/streamlit_app.py              # choose "Annotate golden set"
python scripts/check_golden_set.py              # validates completeness; must pass
```

Follow `data/golden/annotation_guide.md` while labeling. Output:
`data/golden/golden_set.csv`. **This step cannot be automated** — using
model predictions as golden labels would invalidate every downstream metric
(see `docs/decision_log.md` #13).

## 9. Training

```bash
python scripts/05_train.py       # chronological split, trains both classifiers
python scripts/06_build_index.py # builds hybrid retrieval index (TRAIN split only)
```

## 10. Running evaluation

```bash
python scripts/08_evaluate.py         # Baseline A, B, Main System vs golden set
python scripts/09_run_llm_judge.py    # needs OPENAI_API_KEY; else writes PENDING status
python scripts/09b_human_agreement.py # needs data/golden/human_judge_scores.csv; else PENDING
python scripts/10_analyze_failures.py
python scripts/12_leakage_check.py
python scripts/14_run_metadata.py
python scripts/11_generate_report.py
python scripts/13_quality_gate.py
```

Or all at once:

```bash
make evaluate
```

## 11. Running the demo

```bash
streamlit run app/streamlit_app.py
```

Choose "Demo" mode. Shows brand, predicted intent + confidence + top-3,
retrieved historical conversations with scores, draft reply,
grounded/not-grounded, AUTO-HANDLE/ESCALATE + reason.

You can also run the agent from the CLI:

```bash
python scripts/07_run_agent.py "My order hasn't arrived and tracking hasn't updated in 5 days."
```

## 12. Results

See `outputs/report/final_report.md` Section 10 (regenerated by
`scripts/11_generate_report.py` from whatever is currently in
`outputs/metrics/*.json`). **PENDING real-dataset evaluation** — see
Limitations.

## 13. Baselines

- **Baseline A**: majority-intent classifier + fixed generic reply + always
  escalate.
- **Baseline B**: TF-IDF + Logistic Regression intent classifier +
  nearest-neighbor (TF-IDF) historical-reply retrieval + confidence-threshold
  escalation.

Both implemented in `src/evaluation/baselines.py`, evaluated identically to
the main system in `scripts/08_evaluate.py`.

## 14. Failure analysis

`outputs/examples/failure_analysis.csv` (after running the pipeline against
a real golden set) — top-5 real failure categories with representative
examples, generated by `scripts/10_analyze_failures.py` /
`src/evaluation/error_analysis.py`. Never contains invented examples.

## 15. Judge-human agreement

`outputs/metrics/judge_human_agreement.json` +
`outputs/figures/judge_vs_human.png`, produced by
`scripts/09b_human_agreement.py`. **PENDING human annotation of 50 examples**
(`data/golden/human_judge_scores.csv`) — see Section 8/10 above and
`docs/evaluation_protocol.md` Section 6.

## 16. Headline metric

Recommended: **golden-set Macro-F1** (intent classification) alongside
**Safe Automation Rate** (automation rate conditioned on a near-zero
`unsafe_auto_handling_rate`) for the end-to-end system. Full justification:
`outputs/report/final_report.md` Section 10 / `docs/evaluation_protocol.md`
Section 9.

## 17. What is misleading about the headline metric?

Full discussion in `outputs/report/final_report.md` Section 14 — covers
small golden-set size, single-brand scope, temporal distribution shift,
label subjectivity, LLM judge bias, inconsistent historical replies, and why
automation rate and macro-F1 must always be read together with their
safety/per-class counterparts, never alone.

## 18. Limitations

- **No internet access in the build/verification environment** meant the
  real Kaggle dataset could not be downloaded or evaluated against. The
  synthetic demo dataset (`scripts/00b_make_synthetic_dataset.py`) proves
  the pipeline runs correctly end-to-end (see `outputs/metrics/` for real,
  if synthetic-sourced, numbers), but **all headline results on the real
  dataset are PENDING** — run `python scripts/run_all.py --full` after
  setting `DATA_PATH`.
- **Golden-set human annotation is PENDING** — required before any reported
  metric is trustworthy; see Section 8.
- **LLM judge and human-agreement evaluation are PENDING** an
  `OPENAI_API_KEY` and human annotation time respectively; both scripts
  write explicit `PENDING_*` statuses rather than fabricated scores.
- **The intent taxonomy naming step is a stand-in on synthetic data** — for
  the demo run, `scripts/05_train.py` uses a keyword-based weak-labeler
  (`_weak_label_from_keywords`) instead of a human reviewing real cluster
  output, purely so the synthetic pipeline can be exercised end-to-end. On
  real data, replace this with the human-reviewed cluster→intent mapping
  (`src/intents/label.py`).
- **Optional dependencies degrade gracefully but are recommended for full
  quality**: `sentence-transformers` (semantic embeddings — falls back to
  TF-IDF), `rank-bm25` (falls back to a small built-in pure-Python BM25
  implementation with identical scoring), `pyarrow` (parquet I/O — falls
  back to CSV via `src/utils/io.py`). Install them via `requirements.txt`
  for best results; the pipeline still runs without them.
- **Escalation high-risk keyword list is a starting point**
  (`configs/config.yaml` `escalation.high_risk_keywords`), not
  exhaustively tuned against real failure data yet.

## 19. Next steps

See `outputs/report/final_report.md` Section 15, "What I'd Do With One More
Week."

---

## Fast demo mode vs. full run

**FAST DEMO** (bundled synthetic data, <15 minutes total, no dataset
download, no API key needed):

```bash
python scripts/run_all.py --fast
```

This generates the synthetic dataset (if not already present), runs the
full pipeline against it, and produces real metrics/figures/artifacts under
`outputs/` — clearly synthetic-sourced, useful for verifying the pipeline
mechanically works.

**FULL** (real Kaggle data, real results — requires the dataset download and
the human annotation steps in Section 8):

```bash
# after downloading the dataset and setting DATA_PATH in .env, and after
# completing golden-set annotation:
python scripts/run_all.py --full
```

## Repository structure

See the project tree at the bottom of this README, or browse directly —
every `src/` module has a module-level docstring explaining its role, and
every `scripts/*.py` has a `Usage:` comment at the top.

## Tests

```bash
pytest -q
```

Covers: data loading/schema validation, thread reconstruction, text
preprocessing, classifier prediction, retrieval, escalation policy,
response schema, fallback behavior, and end-to-end pipeline execution
(`tests/`).

## License

