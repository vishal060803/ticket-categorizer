from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from main import (
    DEFAULT_REVIEW_THRESHOLD,
    ValidationError,
    predict_ticket,
    preprocess_rows,
    read_csv_rows,
    train_phase3_models,
)


@st.cache_resource(show_spinner=False)
def train_bundle(csv_path: str, use_bigrams: bool) -> Dict[str, Any]:
    rows = read_csv_rows(csv_path)
    processed = preprocess_rows(rows)
    return train_phase3_models(processed_rows=processed, use_bigrams=use_bigrams)


def main() -> None:
    st.set_page_config(page_title="Ticket Categorizer Demo", page_icon="📩", layout="centered")

    st.title("📩 Auto Ticket Categorizer")
    st.caption("Streamlit demo for category prediction + confidence + review routing")

    with st.sidebar:
        st.header("Model Settings")
        data_path = st.text_input("Dataset path", value="tickets.csv")
        use_bigrams = st.checkbox("Use TF-IDF bigrams", value=False)
        review_threshold = st.slider(
            "Human review threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_REVIEW_THRESHOLD),
            step=0.01,
        )

    data_file = Path(data_path)
    if not data_file.exists():
        st.error(f"Dataset not found: {data_file}")
        st.info("Run `python seed_data.py` first or provide the correct dataset path.")
        return

    try:
        with st.spinner("Training models and selecting best one..."):
            bundle = train_bundle(str(data_file), use_bigrams)
    except (ValidationError, FileNotFoundError, RuntimeError) as error:
        st.error(f"Failed to prepare model: {error}")
        return

    st.success(f"Best model: {bundle['best_model_name']}")

    st.subheader("Live Ticket Prediction")
    subject = st.text_input("Subject", placeholder="e.g., Server down for all users")
    body = st.text_area(
        "Body",
        placeholder="Describe the issue, request, or question in detail...",
        height=140,
    )

    if st.button("Predict Category", type="primary"):
        try:
            result = predict_ticket(
                trained_bundle=bundle,
                subject=subject,
                body=body,
                review_threshold=review_threshold,
            )
        except ValidationError as error:
            st.error(str(error))
            return

        st.markdown("### Prediction Result")
        st.write(f"**Predicted Category:** {result['predicted_category']}")
        st.write(f"**Confidence:** {result['confidence_percent']}%")
        st.write(f"**Priority Tag:** {result['priority_tag']}")

        if result["needs_human_review"]:
            st.warning("Needs Human Review: confidence below threshold")
        else:
            st.success("Auto-assign safe: confidence meets threshold")

        with st.expander("Cleaned text used by model"):
            st.code(result["text_clean"])

    st.divider()
    st.caption("Run locally: `streamlit run app.py`")


if __name__ == "__main__":
    main()
