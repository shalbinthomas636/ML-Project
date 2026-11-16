
import streamlit as st
import plotly.express as px

from app.styles import page_header, metric_card
from app.common import format_inr, get_price_column


def render(df, comparison, metadata):
    page_header(
        "Used Car Intelligence Dashboard",
        "A complete overview of used car prices, market data and machine-learning insights."
    )

    price_col = get_price_column(df)

    if price_col is None:
        st.error("The dataset does not contain a selling price column.")
        return

    prices = df[price_col].dropna()
    average = prices.mean()
    highest = prices.max()
    lowest = prices.min()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL VEHICLES", f"{len(df):,}", "Available dataset records")
    with c2:
        metric_card("AVERAGE PRICE", format_inr(average), "Average listed price")
    with c3:
        metric_card("HIGHEST PRICE", format_inr(highest), "Maximum dataset price")
    with c4:
        metric_card("CAR BRANDS", str(df["brand"].nunique()) if "brand" in df else "N/A",
                    "Unique brands")

    st.markdown('<div class="section-heading">Market Overview</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df, x=price_col, nbins=35,
            title="Used Car Price Distribution",
            labels={price_col: "Selling Price (INR)"},
            color_discrete_sequence=["#2589E8"]
        )
        fig.update_layout(template="plotly_white", height=370)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "brand" in df.columns:
            brand_data = (
                df.groupby("brand")[price_col]
                .median().nlargest(10).reset_index()
            )
            fig = px.bar(
                brand_data, x=price_col, y="brand",
                orientation="h",
                title="Top 10 Brands by Median Price",
                labels={price_col: "Median Price", "brand": "Brand"},
                color=price_col,
                color_continuous_scale="Blues"
            )
            fig.update_layout(template="plotly_white", height=370,
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-heading">Vehicle Insights</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        if "fuel" in df.columns:
            fuel_data = df["fuel"].value_counts().reset_index()
            fuel_data.columns = ["Fuel", "Count"]
            fig = px.pie(
                fuel_data, names="Fuel", values="Count",
                title="Vehicles by Fuel Type", hole=0.48,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if "year" in df.columns:
            year_data = (
                df.groupby("year")[price_col]
                .median().reset_index()
            )
            fig = px.line(
                year_data, x="year", y=price_col,
                title="Median Price by Manufacturing Year",
                markers=True,
                labels={"year": "Year", price_col: "Median Price"},
            )
            fig.update_traces(line_color="#1766B5")
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-heading">Machine Learning Status</div>',
                unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Selected algorithm**")
        st.subheader(metadata.get("best_algorithm", "Not available"))
        st.caption("Algorithm selected by your training process")
        st.markdown('</div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Algorithms evaluated**")
        st.subheader(str(len(comparison)) if not comparison.empty else "Not available")
        st.caption("Models recorded in the comparison results")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Dataset Preview</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
