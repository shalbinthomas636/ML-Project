
import streamlit as st
import plotly.express as px

from app.styles import page_header


def render(comparison, metadata):
    page_header(
        "Machine Learning Comparison",
        "Compare regression algorithms and identify the best-performing model."
    )

    if comparison.empty:
        st.warning(
            "Model comparison file was not found. "
            "Run python train_model.py to generate the comparison results."
        )
        return

    df = comparison.copy()
    df.columns = df.columns.str.strip()

    st.markdown("### Algorithm Performance")

    st.dataframe(df, use_container_width=True, hide_index=True)

    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    rmse_col = next(
        (normalized[k] for k in ["rmse", "root_mean_squared_error"]
         if k in normalized), None
    )

    r2_col = next(
        (normalized[k] for k in ["r2", "r2_score", "r_squared"]
         if k in normalized), None
    )

    algorithm_col = next(
        (normalized[k] for k in ["algorithm", "model", "model_name"]
         if k in normalized), df.columns[0]
    )

    best = metadata.get("best_algorithm", "Not available")

    st.markdown("### Selected Best Algorithm")
    st.success(f"Best model from training: {best}")

    if rmse_col:
        chart_data = df.sort_values(rmse_col, ascending=True)
        fig = px.bar(
            chart_data,
            x=algorithm_col,
            y=rmse_col,
            title="RMSE Comparison – Lower is Better",
            color=rmse_col,
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Algorithm",
            yaxis_title="RMSE",
            xaxis_tickangle=-35
        )
        st.plotly_chart(fig, use_container_width=True)

    if r2_col:
        chart_data = df.sort_values(r2_col, ascending=False)
        fig = px.bar(
            chart_data,
            x=algorithm_col,
            y=r2_col,
            title="R² Score Comparison – Higher is Better",
            color=r2_col,
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_tickangle=-35
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Understanding the Metrics")

    with st.expander("What do these evaluation metrics mean?"):
        st.write("""
        - **MAE:** Average absolute difference between actual and predicted prices.
        - **MSE:** Average squared prediction error.
        - **RMSE:** Square root of MSE; expressed in the same unit as the price.
        - **R² Score:** Indicates how much variation in prices is explained by the model.
        - **MAPE:** Average percentage prediction error, where defined.

        A lower MAE, MSE, RMSE and MAPE is generally better.
        A higher R² score is generally better.
        """)
