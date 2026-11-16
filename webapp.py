
import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_car_price_model.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)

DATA_PATH = os.path.join(
    OUTPUT_DIR,
    "cleaned_car_data.csv"
)

COMPARISON_PATH = os.path.join(
    OUTPUT_DIR,
    "algorithm_comparison.csv"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "prediction_history.csv"
)

app = Flask(__name__)

app.secret_key = "autoval-local-development-key"


# --------------------------------------------------
# LOAD APPLICATION DATA
# --------------------------------------------------

def load_resources():

    required = [
        MODEL_PATH,
        METADATA_PATH,
        DATA_PATH,
        COMPARISON_PATH
    ]

    missing = [
        path for path in required
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing. Run train_model.py first: "
            + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    df = pd.read_csv(DATA_PATH)

    comparison = pd.read_csv(COMPARISON_PATH)

    return model, metadata, df, comparison


MODEL = None
METADATA = None
DF = None
COMPARISON = None

try:

    MODEL, METADATA, DF, COMPARISON = load_resources()

    print("AUTOVAL resources loaded successfully.")

except Exception as error:

    print("RESOURCE ERROR:", error)


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def resources_available():

    return all(
        item is not None
        for item in [
            MODEL,
            METADATA,
            DF,
            COMPARISON
        ]
    )


def money(value):

    try:
        return f"₹{float(value):,.0f}"
    except (ValueError, TypeError):
        return "—"


def get_history():

    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_PATH).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_history(record):

    history = get_history()

    new_row = pd.DataFrame([record])

    history = pd.concat(
        [history, new_row],
        ignore_index=True
    )

    history.to_csv(
        HISTORY_PATH,
        index=False
    )


def format_price(value):

    return money(value)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():

    total_cars = len(DF) if DF is not None else 0

    return render_template(
        "index.html",
        total_cars=total_cars,
        ready=resources_available()
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None
    submitted = {}

    if not resources_available():

        flash(
            "Model files are not available. Run train_model.py first.",
            "error"
        )

        return render_template(
            "prediction.html",
            result=None,
            submitted={}
        )

    if request.method == "POST":

        submitted = request.form.to_dict()

        try:

            feature_columns = METADATA["feature_columns"]

            input_data = {}

            for col in feature_columns:

                if col in submitted:

                    value = submitted[col]

                elif col in METADATA["numeric_features"]:

                    value = METADATA["numeric_defaults"].get(
                        col,
                        0
                    )

                else:

                    value = ""

                if col in METADATA["numeric_features"]:

                    if value == "":
                        value = METADATA["numeric_defaults"].get(
                            col,
                            0
                        )

                    input_data[col] = float(value)

                else:

                    input_data[col] = str(value)

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            predicted_price = float(
                MODEL.predict(input_df)[0]
            )

            predicted_price = max(
                predicted_price,
                0
            )

            result = {
                "price": predicted_price,
                "formatted_price": money(predicted_price),
                "algorithm": METADATA["best_algorithm"]
            }

            save_history({

                "date": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                ),

                "brand": submitted.get("brand", ""),

                "name": submitted.get("name", ""),

                "year": submitted.get("year", ""),

                "fuel": submitted.get("fuel", ""),

                "km_driven": submitted.get("km_driven", ""),

                "predicted_price": predicted_price

            })

        except Exception as error:

            flash(
                f"Prediction failed: {error}",
                "error"
            )

    return render_template(
    "prediction.html",
    result=result,
    submitted=submitted,
    metadata=METADATA
)


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

@app.route("/analytics")
def analytics():

    if DF is None:

        return render_template(
            "analytics.html",
            total_cars=0,
            average_price="—",
            average_mileage="—",
            brand_count=0,
            fuel_data={},
            brand_data={},
            year_data={}
        )

    average_price = (
        DF["selling_price"].mean()
        if "selling_price" in DF.columns
        else 0
    )

    average_mileage = (
        DF["mileage"].mean()
        if "mileage" in DF.columns
        else 0
    )

    brand_count = (
        DF["brand"].nunique()
        if "brand" in DF.columns
        else 0
    )

    fuel_data = {}

    if "fuel" in DF.columns:

        fuel_data = (
            DF["fuel"]
            .value_counts()
            .to_dict()
        )

    brand_data = {}

    if "brand" in DF.columns:

        brand_data = (
            DF.groupby("brand")["selling_price"]
            .mean()
            .nlargest(10)
            .round(0)
            .to_dict()
        )

    year_data = {}

    if "year" in DF.columns:

        year_data = (
            DF.groupby("year")["selling_price"]
            .mean()
            .dropna()
            .round(0)
            .to_dict()
        )

    return render_template(
        "analytics.html",
        total_cars=len(DF),
        average_price=money(average_price),
        average_mileage=round(average_mileage, 2),
        brand_count=brand_count,
        fuel_data=fuel_data,
        brand_data=brand_data,
        year_data=year_data
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

@app.route("/comparison")
def comparison():

    if COMPARISON is None:

        return render_template(
            "comparison.html",
            comparison=[],
            best_algorithm=None
        )

    comparison_records = (
        COMPARISON
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_dict(orient="records")
    )

    best_algorithm = (
        comparison_records[0]["Algorithm"]
        if comparison_records
        else None
    )

    return render_template(
        "comparison.html",
        comparison=comparison_records,
        best_algorithm=best_algorithm
    )


# --------------------------------------------------
# EXPLORE VEHICLES
# --------------------------------------------------

@app.route("/explore")
def explore():

    if DF is None:

        return render_template(
            "explore.html",
            cars=[],
            fuel_options=[],
            brand_options=[]
        )

    filtered = DF.copy()

    search = request.args.get(
        "search",
        ""
    ).strip()

    fuel = request.args.get(
        "fuel",
        ""
    ).strip()

    transmission = request.args.get(
        "transmission",
        ""
    ).strip()

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    if search:

        search_columns = [
            col for col in ["brand", "name"]
            if col in filtered.columns
        ]

        if search_columns:

            mask = pd.Series(
                False,
                index=filtered.index
            )

            for col in search_columns:

                mask |= filtered[col].astype(
                    str
                ).str.contains(
                    search,
                    case=False,
                    na=False
                )

            filtered = filtered[mask]

    if fuel and "fuel" in filtered.columns:

        filtered = filtered[
            filtered["fuel"].astype(str) == fuel
        ]

    if transmission and "transmission" in filtered.columns:

        filtered = filtered[
            filtered["transmission"].astype(str)
            == transmission
        ]

    if brand and "brand" in filtered.columns:

        filtered = filtered[
            filtered["brand"].astype(str) == brand
        ]

    fuel_options = (
        sorted(DF["fuel"].dropna().unique().tolist())
        if "fuel" in DF.columns
        else []
    )

    brand_options = (
        sorted(DF["brand"].dropna().unique().tolist())
        if "brand" in DF.columns
        else []
    )

    cars = filtered.head(300).fillna("").to_dict(
        orient="records"
    )

    return render_template(
        "explore.html",
        cars=cars,
        fuel_options=fuel_options,
        brand_options=brand_options
    )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.route("/history")
def history():

    history_df = get_history()

    records = history_df.to_dict(
        orient="records"
    )

    total = len(records)

    if total:

        latest_date = records[-1].get(
            "date",
            "—"
        )

        average_prediction = money(
            pd.to_numeric(
                history_df["predicted_price"],
                errors="coerce"
            ).mean()
        )

    else:

        latest_date = "—"
        average_prediction = "—"

    return render_template(
        "history.html",
        history=records,
        latest_date=latest_date,
        average_prediction=average_prediction
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def page_not_found(error):

    return "Page not found.", 404


@app.errorhandler(405)
def method_not_allowed(error):

    return "This request method is not supported for this page.", 405


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
