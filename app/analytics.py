
import streamlit as st
import pandas as pd
import plotly.express as px

from app.styles import page_header
from app.common import get_price_column


def render(df):
    page_header(
        "Data Analytics",
        "Discover patterns, trends and relationships in used car prices."
    )

    price_col = get_price_column(df)

    if price_col is None:
        st.error("Selling price column is unavailable.")
        return

    numeric = df.select_dtypes(include="number")

    st.markdown("### Correlation Analysis")

    if len(numeric.columns) >= 2:
        correlation = numeric.corr(numeric_only=True)

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Numerical Feature Correlation Heatmap"
        )
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Price Relationship Analysis")

    available = [
        c for c in [
            "year", "km_driven", "mileage",
            "engine", "max_power", "seats"
        ] if c in df.columns
    ]

    if available:
        feature = st.selectbox("Choose a feature", available)

        chart_data = df[[feature, price_col]].dropna()

        fig = px.scatter(
            chart_data,
            x=feature,
            y=price_col,
            opacity=0.55,
            title=f"{feature.replace('_', ' ').title()} vs Selling Price",
            labels={price_col: "Selling Price (INR)"}
        )
        fig.update_traces(marker={"color": "#2589E8"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Brand-Wise Analysis")

    if "brand" in df.columns:
        brand_analysis = (
            df.groupby("brand")[price_col]
            .agg(["count", "mean", "median"])
            .reset_index()
            .sort_values("median", ascending=False)
        )

        brand_analysis.columns = [
            "Brand", "Vehicle Count", "Average Price", "Median Price"
        ]

        st.dataframe(
            brand_analysis,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            brand_analysis.head(15),
            x="Brand",
            y="Median Price",
            title="Top 15 Brands by Median Selling Price",
            color="Median Price",
            color_continuous_scale="Blues"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Fuel Type Analysis")

    if "fuel" in df.columns:
        fuel_analysis = (
            df.groupby("fuel")[price_col]
            .median().reset_index()
        )

        fig = px.bar(
            fuel_analysis,
            x="fuel",
            y=price_col,
            title="Median Selling Price by Fuel Type",
            color=price_col,
            color_continuous_scale="Blues"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)
