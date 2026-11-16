
import streamlit as st
import plotly.express as px

from app.styles import page_header
from app.common import get_price_column, format_inr


def render(df):
    page_header(
        "Explore Used Cars",
        "Search, filter and compare vehicles from the dataset."
    )

    data = df.copy()
    price_col = get_price_column(data)

    if price_col is None:
        st.error("Selling price column is unavailable.")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        if "brand" in data:
            brands = sorted(data["brand"].dropna().astype(str).unique())
            selected_brand = st.selectbox("Select Brand", ["All"] + brands)
        else:
            selected_brand = "All"

    with c2:
        if "fuel" in data:
            fuels = sorted(data["fuel"].dropna().astype(str).unique())
            selected_fuel = st.selectbox("Fuel Type", ["All"] + fuels)
        else:
            selected_fuel = "All"

    with c3:
        if "transmission" in data:
            transmissions = sorted(
                data["transmission"].dropna().astype(str).unique()
            )
            selected_transmission = st.selectbox(
                "Transmission", ["All"] + transmissions
            )
        else:
            selected_transmission = "All"

    minimum = float(data[price_col].min())
    maximum = float(data[price_col].max())

    price_range = st.slider(
        "Selling Price Range (INR)",
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        step=max(1.0, (maximum - minimum) / 100)
    )

    filtered = data[
        data[price_col].between(price_range[0], price_range[1])
    ]

    if selected_brand != "All" and "brand" in filtered:
        filtered = filtered[filtered["brand"].astype(str) == selected_brand]

    if selected_fuel != "All" and "fuel" in filtered:
        filtered = filtered[filtered["fuel"].astype(str) == selected_fuel]

    if selected_transmission != "All" and "transmission" in filtered:
        filtered = filtered[
            filtered["transmission"].astype(str) == selected_transmission
        ]

    search = st.text_input("Search by vehicle name or brand")

    if search:
        searchable = [
            c for c in ["name", "brand"]
            if c in filtered.columns
        ]

        if searchable:
            mask = filtered[searchable].astype(str).apply(
                lambda column: column.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)

            filtered = filtered[mask]

    st.markdown(f"### {len(filtered):,} Vehicles Found")

    display_columns = [
        c for c in [
            "name", "brand", "year", price_col,
            "km_driven", "fuel", "transmission",
            "owner", "mileage", "engine", "max_power", "seats"
        ] if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Cars",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_used_cars.csv",
        mime="text/csv"
    )

    st.markdown("### Compare Cars")

    if len(filtered) >= 2:
        options = filtered.index.tolist()

        selected = st.multiselect(
            "Choose two or more cars",
            options=options,
            format_func=lambda idx: (
                str(filtered.loc[idx, "name"])
                if "name" in filtered.columns
                else f"Vehicle {idx}"
            ),
            max_selections=4
        )

        if len(selected) >= 2:
            comparison = filtered.loc[selected, display_columns]
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                comparison.reset_index(),
                x="name" if "name" in comparison.columns else "index",
                y=price_col,
                title="Selected Vehicle Price Comparison",
                color=price_col,
                color_continuous_scale="Blues"
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough vehicles are available for comparison.")
