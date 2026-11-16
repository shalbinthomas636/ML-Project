
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.styles import page_header
from app.common import (
    build_prediction_input,
    get_feature_lists,
    format_inr,
    ROOT
)


HISTORY_PATH = ROOT / "outputs" / "prediction_history.csv"


def save_prediction(details, prediction, algorithm):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **details,
        "predicted_price": float(prediction),
        "algorithm": algorithm
    }

    new_data = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        old_data = pd.read_csv(HISTORY_PATH)
        new_data = pd.concat([old_data, new_data], ignore_index=True)

    new_data.to_csv(HISTORY_PATH, index=False)


def render(model, metadata, df):
    page_header(
        "Car Price Prediction",
        "Enter your vehicle specifications and receive a machine-learning price estimate."
    )

    features, numeric_features, categorical_features = get_feature_lists(metadata)

    defaults = metadata.get("numeric_defaults", {})
    category_options = metadata.get("category_options", {})

    dataset_numeric = {}
    for feature in numeric_features:
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce").dropna()
            if not values.empty:
                dataset_numeric[feature] = values

    st.markdown("### Vehicle Specifications")
    st.caption("Complete the fields below. The prediction uses the same features as your trained model.")

    values = {}

    with st.form("car_prediction_form"):
        left, right = st.columns(2)

        for index, feature in enumerate(features):
            container = left if index % 2 == 0 else right

            with container:
                if feature in categorical_features:
                    options = category_options.get(feature, [])

                    if not options and feature in df.columns:
                        options = sorted(
                            df[feature].dropna().astype(str).unique().tolist()
                        )

                    options = [str(x) for x in options if str(x).strip()]

                    if options:
                        default = options[0]
                        values[feature] = st.selectbox(
                            feature.replace("_", " ").title(),
                            options,
                            index=0,
                            key=f"prediction_{feature}"
                        )
                    else:
                        values[feature] = st.text_input(
                            feature.replace("_", " ").title(),
                            value="Unknown",
                            key=f"prediction_{feature}"
                        )

                elif feature in numeric_features:
                    series = dataset_numeric.get(feature)
                    default = float(defaults.get(feature, 0) or 0)

                    if series is not None and not series.empty:
                        minimum = float(series.min())
                        maximum = float(series.max())
                        default = min(max(default, minimum), maximum)
                    else:
                        minimum = 0.0
                        maximum = max(default * 3, 100.0)

                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=1.0,
                        key=f"prediction_{feature}"
                    )
                else:
                    values[feature] = st.text_input(
                        feature.replace("_", " ").title(),
                        value=str(defaults.get(feature, "")),
                        key=f"prediction_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict Car Selling Price",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = build_prediction_input(values, metadata)
            prediction = float(model.predict(input_df)[0])

            if prediction < 0:
                st.error("The model returned a negative price. Please check the input values.")
                return

            st.markdown(f"""
            <div class="price-result">
                <div>ESTIMATED SELLING PRICE</div>
                <div class="amount">{format_inr(prediction)}</div>
                <div>Machine-learning prediction</div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Price prediction generated successfully.")

            algorithm = metadata.get("best_algorithm", "Best Model")
            save_prediction(values, prediction, algorithm)

            st.markdown("### Prediction Details")
            result_df = pd.DataFrame({
                "Vehicle Feature": list(values.keys()),
                "Entered Value": [str(v) for v in values.values()]
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            report = pd.DataFrame([{
                **values,
                "predicted_price": prediction,
                "algorithm": algorithm
            }])

            st.download_button(
                "Download Prediction Report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="car_price_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.info(
                "This is an estimated price produced by the trained model, "
                "not a guaranteed market selling price."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)
