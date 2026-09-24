# Decision Log

Non-obvious engineering decisions made while building HIVER_SUPPORT_AGENT,
why, what was considered instead, and the trade-off accepted.

---

### 1. Selecting a single brand via a composite volume/completeness score, not just message count
**Why:** Raw message count favors brands with lots of one-off inbound chatter
but few actual resolved threads, which is what retrieval and training need.
**Alternative considered:** Rank by total tweet count per author.
**Trade-off:** The chosen brand may not be the single largest account overall,
but it has the most usable (customer→resolution) pairs, which is what the
agent is actually built on.

### 2. Chronological train/val/test split instead of random split
**Why:** Support conversations have topic drift over time (new features, new
bugs); a random split lets the model "see the future" implicitly through
near-duplicate contemporaneous tickets. Chronological split resembles real
production deployment, where the model only ever sees past resolutions.
**Alternative considered:** Stratified random split by intent.
**Trade-off:** Class balance across time can be uneven (e.g. an intent that
only appears in a later period is under-represented in train), which is
visible in per-intent support counts in the classification report.

### 3. Golden set held out entirely, never touched by training or retrieval
**Why:** Required for honest evaluation — the agent must never have
"studied" the exact examples it's scored on, either as classifier training
data or as retrievable historical evidence.
**Alternative considered:** Allow golden examples with matching intents into
the retrieval index (common in some RAG setups) since it's "just evidence."
**Trade-off:** Slightly less retrieval coverage, but eliminates the leakage
vector the automated leakage check (`scripts/12_leakage_check.py`) exists to
catch.

### 4. 6–10 intents derived from clustering, not a fixed pre-written taxonomy
**Why:** The assignment requires the taxonomy to reflect what this brand's
customers actually write about, not a generic textbook list.
**Alternative considered:** Ship a fixed 8-intent list up front (faster).
**Trade-off:** Requires a human review step after clustering
(`scripts/03_discover_intents.py`) before training can safely proceed;
slower but avoids a taxonomy that doesn't match the data.

### 5. Explicit `OTHER_OR_UNKNOWN` intent with a confidence threshold, not
forced top-1 classification
**Why:** Forcing every message into a fixed intent produces confidently
wrong labels on genuinely novel/ambiguous messages, which is dangerous for
an auto-handling system.
**Alternative considered:** Always predict the argmax intent.
**Trade-off:** Slightly lower raw accuracy in exchange for a mechanism that
correctly triggers escalation instead of guessing.

### 6. Hybrid retrieval weighting (0.4 lexical / 0.6 semantic), not lexical- or
semantic-only
**Why:** Lexical (BM25) catches exact entity/keyword matches (order numbers,
product names) that embeddings can blur together; semantic similarity
catches paraphrases lexical search misses. Neither alone was sufficient in
early manual spot-checks.
**Alternative considered:** Semantic-only (simpler pipeline).
**Trade-off:** More moving parts (two retrievers to maintain), but
meaningfully more robust retrieval, especially for entity-heavy tickets
(order IDs, exact dollar amounts).

### 7. Classifier confidence threshold (0.60) selected via a validation-set
precision/coverage trade-off, not to maximize a single metric
**Why:** A threshold chosen purely to maximize accuracy or F1 can silently
push the system toward over-automation. The threshold should be chosen so
that AUTO_HANDLE decisions above it have acceptably high precision on the
validation split.
**Alternative considered:** Pick the threshold that maximizes macro-F1 on
validation.
**Trade-off:** 0.60 is a reasonable, inspectable starting point but is NOT
re-derived from real Kaggle-brand validation data in this repository (no
real dataset was available in this environment) — this is explicitly listed
as a pending real-data step. See README "Limitations."

### 8. Escalation policy is a prioritized rule list, not a single ML classifier
**Why:** Escalation is a safety-critical decision. A prioritized, inspectable
rule list (`src/escalation/policy.py`) lets every decision be explained with
one specific reason code, which a learned classifier's probability score
cannot do as cleanly, and which is easier for a human reviewer to audit.
**Alternative considered:** Train a binary escalate/auto-handle classifier on
weak labels.
**Trade-off:** Rules require manual tuning/keyword lists (e.g. high-risk
keywords) rather than being learned automatically, but the resulting
decisions are fully explainable, which matters more here than marginal
accuracy gains.

### 9. LLM-free fallback returns the best retrieved historical reply as-is,
not an auto-edited/templated version
**Why:** Any automatic substitution of entities (order IDs, amounts) without
an LLM risks inserting incorrect specifics with no verification step. Using
the historical reply verbatim, gated on a minimum similarity score, is the
safest fallback that still produces a real, historically-grounded answer.
**Alternative considered:** Regex-based templating (replace order numbers
with the current customer's number).
**Trade-off:** Fallback replies sometimes still reference the wrong specific
order ID from the retrieved example; this is intentional and documented as a
known limitation (see failure analysis) rather than silently "fixed" in a
way that could fabricate correctness.

### 10. BLEU/ROUGE explicitly NOT used as the primary reply-quality metric
**Why:** Both measure n-gram overlap with a single reference and correlate
poorly with whether a reply is actually correct, safe, or helpful — a
reply can score well by copying phrasing while still making an unsupported
claim, or score poorly while being a perfectly good paraphrase.
**Alternative considered:** Report BLEU/ROUGE as headline metrics since
they're cheap and require no API key.
**Trade-off:** The chosen alternative (LLM judge + human agreement +
groundedness heuristics) costs more to compute and requires human labeling
time, but actually measures what the assignment cares about.

### 11. LLM-as-judge scores are never treated as ground truth without a human
agreement check
**Why:** An unvalidated LLM judge can systematically over- or under-score in
ways that look precise but aren't trustworthy. The assignment explicitly
requires evidence the judge tracks human judgment.
**Alternative considered:** Report judge scores directly as the reply-quality
headline number.
**Trade-off:** Adds a whole extra evaluation stage
(`scripts/09b_human_agreement.py`, `data/golden/human_judge_scores.csv`)
that requires real human labeling time before judge scores can be trusted
in the report.

### 12. Human-agreement sample size fixed at 50 (subset of the 200 golden
examples), not the full golden set
**Why:** Full-golden-set human judge-rubric labeling (200 examples × 6
dimensions) is a large time cost for marginal statistical benefit over a
well-chosen 50-example subset; 50 is enough to estimate agreement
statistics (weighted kappa, correlation) with reasonable confidence
intervals.
**Alternative considered:** Score all 200 examples by hand.
**Trade-off:** Slightly wider confidence intervals on agreement statistics,
in exchange for a much more feasible annotation workload.

### 13. Weak/cluster-derived intent labels are used only for classifier
TRAINING, never as golden evaluation labels
**Why:** Using model-derived or heuristic labels as "ground truth" for
evaluation would make every downstream metric circular and meaningless —
this is explicitly forbidden by the assignment.
**Alternative considered:** Bootstrap the golden set from high-confidence
classifier predictions, then spot-check a sample (faster than full manual
labeling).
**Trade-off:** Full manual labeling of all golden examples is slower but is
the only way the reported metrics are honest.

### 14. Synthetic demo dataset shipped for pipeline verification, sharply
separated from real evaluation results
**Why:** This repository's build/verification environment lacked internet
access to download the real Kaggle dataset. Rather than leaving the pipeline
unverified, a schema-matched synthetic dataset generator
(`scripts/00b_make_synthetic_dataset.py`) lets every stage run end-to-end
and produces real (if synthetic-sourced) numbers, clearly labeled as such
everywhere they appear.
**Alternative considered:** Ship the pipeline unverified / only unit-tested.
**Trade-off:** Higher confidence the mechanical pipeline actually works, at
the cost of needing to repeatedly flag "this number is from synthetic demo
data, not the real dataset" throughout outputs and the report.

### 15. Data leakage checks re-verify overlap at multiple pipeline stages
(golden vs. train, cross-split duplicates, golden vs. retrieval index)
rather than trusting the split logic once
**Why:** Each of these is a genuinely different leakage vector (training
signal leakage, evaluation-set contamination, retrieval-time leakage) that
could be reintroduced independently by a future code change; checking all
three catches regressions the others wouldn't.
**Alternative considered:** Single leakage check at split time only.
**Trade-off:** More code/maintenance, but this check caught a real bug
during this project's own development (see README "Limitations" — an
ad-hoc smoke-test golden set built after index construction correctly
tripped `golden_not_in_retrieval_index`), which is direct evidence the
check works.
