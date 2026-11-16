
import streamlit as st
import pandas as pd

from app.styles import page_header
from app.common import ROOT, format_inr

HISTORY_PATH = ROOT / "outputs" / "prediction_history.csv"


def render():
    page_header(
        "Prediction History",
        "Review and download your previous car price estimates."
    )

    if not HISTORY_PATH.exists():
        st.info("No predictions have been saved yet. Make your first prediction.")
        return

    try:
        history = pd.read_csv(HISTORY_PATH)
    except Exception as error:
        st.error(f"Unable to read prediction history: {error}")
        return

    if history.empty:
        st.info("Prediction history is empty.")
        return

    st.markdown("### Prediction Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Total Predictions", len(history))

    with c2:
        if "predicted_price" in history:
            st.metric(
                "Average Predicted Price",
                format_inr(history["predicted_price"].mean())
            )

    st.markdown("### Previous Predictions")

    st.dataframe(
        history.sort_values(
            "date", ascending=False
        ) if "date" in history else history,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Complete History",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    st.warning(
        "Clearing history permanently deletes the local prediction history CSV."
    )

    if st.button("Clear Prediction History"):
        HISTORY_PATH.unlink(missing_ok=True)
        st.success("Prediction history cleared.")
        st.rerun()
