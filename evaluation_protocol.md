# Evaluation Protocol

This document specifies exactly how evaluation numbers in this repository
are produced, so any reported metric is independently reproducible.

## 1. Golden set

- Size: 150–250 examples (target 200), sampled by
  `scripts/04_create_golden_candidates.py`, stratified across message-length
  bucket (short/medium/long) and time-period quartile.
- Labeling: 100% human, via `streamlit run app/streamlit_app.py` → "Annotate
  golden set" mode, following `data/golden/annotation_guide.md`. Output:
  `data/golden/golden_set.csv`.
- Validation: `python scripts/check_golden_set.py` must exit 0 before any
  headline number is reported. It checks: no duplicate IDs/texts, all
  required fields present, intent values valid against
  `configs/intents.yaml`, escalation reason codes valid, row count within
  150–250, and warns on any intent with <3 golden examples.
- The golden set is excluded from classifier training, from the retrieval
  index, and from train/val/test splits (`src/data/split_data.py`
  `golden_customer_texts` parameter).

## 2. Baselines

- **Baseline A (trivial)**: majority-class intent, fixed generic reply,
  always escalates. Establishes the floor.
- **Baseline B (simple)**: TF-IDF + Logistic Regression intent classifier;
  reply = nearest-neighbor (TF-IDF cosine) historical response; escalation =
  simple confidence threshold rule. Establishes a credible classical
  baseline.
- **Main system**: semantic classifier + hybrid retrieval + grounded
  generation (LLM or fallback) + explicit escalation policy.

All three are run against the identical golden set in the same
`scripts/08_evaluate.py` invocation for a fair, matched comparison.

## 3. Classification metrics

Computed by `src/intents/evaluate.py::compute_classification_metrics`:
accuracy, macro-F1 (primary — treats all intents equally regardless of
frequency, so rare-but-important intents aren't hidden by common ones),
weighted-F1, and per-intent precision/recall/F1/support. Confusion matrices
are plotted to `outputs/figures/{system}_confusion_matrix.png`.

## 4. Escalation metrics

Computed by `src/evaluation/escalation_metrics.py`, treating escalation as
binary classification (`AUTO_HANDLE` vs `ESCALATE`) against
`gold_escalate`. Reports precision/recall/F1 on the ESCALATE class, a full
confusion matrix, `auto_handle_accuracy` (of examples the system chose to
auto-handle, what fraction were gold-labeled AUTO_HANDLE), and the
safety-critical `unsafe_auto_handling_rate` = P(gold=ESCALATE AND
predicted=AUTO_HANDLE). This is the single most important safety number in
the whole evaluation.

## 5. Reply-quality: LLM-as-judge

`scripts/09_run_llm_judge.py` scores every main-system golden-set reply on
groundedness / relevance / correctness / helpfulness / tone / overall (1–5
each), using the rubric in `src/evaluation/llm_judge.py`. Requires
`OPENAI_API_KEY`; if unset, writes `PENDING_NO_OPENAI_API_KEY` rather than a
fabricated score.

## 6. Human agreement with the LLM judge

`scripts/09b_human_agreement.py` requires a human to independently score a
50-example subset (same rubric, same examples) and save to
`data/golden/human_judge_scores.csv`. Agreement is computed per-dimension:
exact agreement rate, mean absolute difference, Pearson correlation, and
linear-weighted Cohen's kappa (`src/evaluation/human_agreement.py`). Without
this file, the script reports `PENDING_HUMAN_ANNOTATION` with exact
instructions — never a fabricated agreement number.

## 7. Failure analysis

`scripts/10_analyze_failures.py` runs rule-based categorization
(`src/evaluation/error_analysis.py`) over the real golden-set evaluation
rows produced by step 8 (`outputs/examples/main_system_eval_rows.csv`),
extracting the top-5 failure categories by frequency with up to 3 real
representative examples each. No example is invented.

## 8. Leakage checks

`scripts/12_leakage_check.py` verifies, using the actual files on disk
(not assumptions): golden texts absent from train split, no duplicate texts
across train/val/test, golden texts absent from the built retrieval index,
and that `golden_set.csv` contains no `pred_*` columns (i.e. labels were not
back-filled from model predictions). Status is `PASS`, `FAIL`, or
`PARTIAL_PENDING` (some checks not yet runnable because a prior artifact
doesn't exist yet) — never silently skipped.

## 9. Headline metric

See `outputs/report/final_report.md` section "Headline Number" for the
chosen primary metric and full justification, and "What Is Misleading About
My Headline Number?" for its explicit limitations.

## 10. Reproducing all of the above

```
python scripts/run_all.py --fast   # synthetic demo data, <15 minutes
python scripts/run_all.py --full   # real Kaggle data, after DATA_PATH is set
                                     # and the golden set has been annotated
```

Every metric file under `outputs/metrics/*.json` records either a real
number or an explicit `PENDING_*` status with the exact command to complete
it — never a fabricated value.
