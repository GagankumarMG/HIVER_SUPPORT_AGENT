"""AI Customer Support Agent - Streamlit app.

Two modes, selectable in the sidebar:
  1. "Demo": run the trained agent pipeline interactively on example or
     custom customer messages.
  2. "Annotate golden set": human annotation UI for data/golden/golden_candidates.csv
     -> data/golden/golden_set.csv (see data/golden/annotation_guide.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.escalation import reason_codes as rc  # noqa: E402
from src.utils.config import load_settings  # noqa: E402

st.set_page_config(page_title="AI Customer Support Agent", layout="wide")


@st.cache_resource
def _load_pipeline():
    from src.agent.pipeline import PipelineConfig, SupportAgentPipeline
    from src.generation.llm_client import LLMClient
    from src.intents.classifier import SemanticClassifier
    from src.retrieval.build_index import load_index

    settings = load_settings()
    classifier = SemanticClassifier.load("models/main_semantic_classifier.joblib")
    retriever = load_index("models/hybrid_retrieval_index.joblib")
    llm_client = LLMClient(settings.openai_api_key, settings.openai_model)
    config = PipelineConfig(
        brand=settings.brand_config.get("brand", {}).get("display_name") or "Brand",
        confidence_threshold=settings.get("classifier", "confidence_threshold", default=0.60),
        min_hybrid_score_for_grounding=settings.get(
            "retrieval", "min_hybrid_score_for_grounding", default=0.35
        ),
        top_k_retrieval=settings.get("retrieval", "top_k", default=5),
        high_risk_keywords=settings.get("escalation", "high_risk_keywords", default=[]),
    )
    return SupportAgentPipeline(classifier, retriever, config, llm_client), settings


def demo_mode():
    st.title("AI Customer Support Agent")
    try:
        pipeline, settings = _load_pipeline()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run the pipeline first: "
            "`python scripts/run_all.py --fast` (or `--full` with real data), "
            "which trains models and builds the retrieval index."
        )
        return

    brand = settings.brand_config.get("brand", {}).get("display_name") or "(brand not yet selected)"
    st.caption(f"Brand: **{brand}**")

    examples = []
    golden_path = Path("data/golden/golden_set.csv")
    if golden_path.exists():
        examples = pd.read_csv(golden_path)["customer_text"].dropna().head(5).tolist()

    with st.sidebar:
        st.header("Try an example")
        chosen = st.selectbox("Example messages", ["(custom)"] + examples)

    default_text = "" if chosen == "(custom)" else chosen
    message = st.text_area("Customer message", value=default_text, height=100)

    if st.button("Run agent", type="primary") and message.strip():
        with st.spinner("Running pipeline..."):
            result = pipeline.run(message)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Draft reply")
            st.info(result.draft_reply)

            st.subheader("Retrieved historical conversations")
            for ex in result.retrieved_examples:
                with st.expander(f"score={ex.score:.2f} — {ex.customer_message[:60]}..."):
                    st.write(f"**Historical customer message:** {ex.customer_message}")
                    st.write(f"**Historical support reply:** {ex.historical_reply}")

        with col2:
            st.subheader("Predicted intent")
            st.metric("Intent", result.intent, delta=f"{result.intent_confidence:.0%} confidence")
            st.write("Top intents:")
            for ti in result.top_intents:
                st.write(f"- {ti.intent}: {ti.score:.2%}")

            st.subheader("Decision")
            if result.decision == "AUTO_HANDLE":
                st.success("AUTO-HANDLE")
            else:
                st.warning(f"ESCALATE — {result.reason_code}")
                st.write(result.escalation_reason)

            st.subheader("Grounding")
            st.write("✅ Grounded" if result.grounded else "⚠️ Not grounded")
            if result.unsupported_claims:
                st.write("Unsupported claims flagged:")
                for c in result.unsupported_claims:
                    st.write(f"- {c}")


def annotation_mode():
    st.title("Golden Set Annotation")
    candidates_path = Path("data/golden/golden_candidates.csv")
    golden_path = Path("data/golden/golden_set.csv")

    if not candidates_path.exists():
        st.error("data/golden/golden_candidates.csv not found. Run "
                  "`python scripts/04_create_golden_candidates.py` first.")
        return

    if golden_path.exists():
        df = pd.read_csv(golden_path)
    else:
        df = pd.read_csv(candidates_path)

    try:
        with open("configs/intents.yaml") as f:
            intents_cfg = yaml.safe_load(f)
        intent_names = [i["name"] for i in intents_cfg.get("intents", [])]
    except FileNotFoundError:
        intent_names = ["OTHER_OR_UNKNOWN"]

    st.markdown("See `data/golden/annotation_guide.md` for full labeling instructions.")

    unlabeled = df[df["gold_intent"].isna() | (df["gold_intent"].astype(str).str.strip() == "")]
    st.caption(f"{len(df) - len(unlabeled)}/{len(df)} labeled")

    if unlabeled.empty:
        st.success("All examples labeled! Run `python scripts/check_golden_set.py` to validate.")
        return

    idx = unlabeled.index[0]
    row = df.loc[idx]

    st.subheader("Customer message")
    st.write(row["customer_text"])
    with st.expander("Full context"):
        st.write(row.get("context", ""))

    col1, col2 = st.columns(2)
    with col1:
        gold_intent = st.selectbox("Intent", intent_names, key=f"intent_{idx}")
        gold_escalate = st.radio("Decision", ["AUTO_HANDLE", "ESCALATE"], key=f"esc_{idx}")
    with col2:
        gold_reason = None
        if gold_escalate == "ESCALATE":
            gold_reason = st.selectbox("Escalation reason", rc.ALL_REASON_CODES, key=f"reason_{idx}")
        gold_quality = st.slider("Best-possible reply quality (1-5)", 1, 5, 3, key=f"qual_{idx}")

    notes = st.text_area("Annotator notes (optional)", key=f"notes_{idx}")

    if st.button("Save & Next", type="primary"):
        df.loc[idx, "gold_intent"] = gold_intent
        df.loc[idx, "gold_escalate"] = gold_escalate
        df.loc[idx, "gold_escalation_reason"] = gold_reason or ""
        df.loc[idx, "gold_reply_quality"] = gold_quality
        df.loc[idx, "annotator_notes"] = notes
        df.to_csv(golden_path, index=False)
        st.rerun()


def main():
    st.sidebar.title("Mode")
    mode = st.sidebar.radio("Select mode", ["Demo", "Annotate golden set"])
    if mode == "Demo":
        demo_mode()
    else:
        annotation_mode()


if __name__ == "__main__":
    main()
