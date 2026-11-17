
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATASET_DIR = ROOT / "dataset"


# ==================================================
# MODEL FILE PATHS
# ==================================================

MODEL_PATH = MODEL_DIR / "best_car_price_model.pkl"

METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ==================================================
# DATASET CANDIDATES
# ==================================================

DATASET_PATHS = [
    OUTPUT_DIR / "cleaned_dataset.csv",
    OUTPUT_DIR / "cleaned_car_data.csv",
    DATASET_DIR / "cleaned_dataset.csv",
    DATASET_DIR / "Car details v3.csv",
    ROOT / "Car details v3.csv",
]


# ==================================================
# COMPARISON FILE CANDIDATES
# ==================================================

COMPARISON_PATHS = [
    OUTPUT_DIR / "model_comparison.csv",
    OUTPUT_DIR / "algorithm_comparison.csv",
    ROOT / "model_comparison.csv",
    ROOT / "algorithm_comparison.csv",
]


# ==================================================
# FIND EXISTING FILE
# ==================================================

def find_existing(paths):

    for path in paths:

        if path.is_file():

            return path

    return None


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run python train_model.py and upload the model file."
        )

    try:

        return joblib.load(MODEL_PATH)

    except Exception as error:

        raise RuntimeError(
            f"Unable to load trained model: {error}"
        ) from error


# ==================================================
# LOAD MODEL METADATA
# ==================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.is_file():

        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        if not isinstance(metadata, dict):

            raise ValueError(
                "Model metadata must contain a JSON object."
            )

        return metadata

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid metadata JSON: {error}"
        ) from error


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_dataset():

    path = find_existing(DATASET_PATHS)

    if path is None:

        raise FileNotFoundError(
            "Dataset not found.\n"
            "Checked locations:\n"
            + "\n".join(str(p) for p in DATASET_PATHS)
        )

    try:

        df = pd.read_csv(path)

        if df.empty:

            raise ValueError("Dataset is empty.")

        df.columns = (
            df.columns.astype(str).str.strip()
        )

        # Generate brand column if missing
        if "brand" not in df.columns and "name" in df.columns:

            df["brand"] = (
                df["name"]
                .fillna("Unknown")
                .astype(str)
                .str.split()
                .str[0]
            )

        # Convert selling price to numeric
        if "selling_price" in df.columns:

            df["selling_price"] = pd.to_numeric(
                df["selling_price"],
                errors="coerce"
            )

        return df

    except Exception as error:

        raise RuntimeError(
            f"Unable to load dataset ({path}): {error}"
        ) from error


# ==================================================
# LOAD MODEL COMPARISON
# ==================================================

@st.cache_data
def load_comparison():

    path = find_existing(COMPARISON_PATHS)

    if path is None:

        return pd.DataFrame()

    try:

        comparison = pd.read_csv(path)

        comparison.columns = (
            comparison.columns.astype(str).str.strip()
        )

        return comparison

    except Exception as error:

        raise RuntimeError(
            f"Unable to load comparison file ({path}): {error}"
        ) from error


# ==================================================
# FEATURE INFORMATION
# ==================================================

def get_feature_lists(metadata):

    numeric = metadata.get(
        "numeric_features", []
    )

    categorical = metadata.get(
        "categorical_features", []
    )

    features = metadata.get(
        "feature_columns",
        numeric + categorical
    )

    return features, numeric, categorical


# ==================================================
# BUILD PREDICTION INPUT
# ==================================================

def build_prediction_input(values, metadata):

    features, numeric, categorical = (
        get_feature_lists(metadata)
    )

    row = {}

    for feature in features:

        if feature in values:

            row[feature] = values[feature]

        elif feature in metadata.get(
            "numeric_defaults", {}
        ):

            row[feature] = (
                metadata["numeric_defaults"][feature]
            )

        elif feature in metadata.get(
            "category_options", {}
        ):

            options = metadata["category_options"][feature]

            row[feature] = (
                options[0] if options else "Unknown"
            )

        else:

            row[feature] = 0

    return pd.DataFrame(
        [row],
        columns=features
    )


# ==================================================
# FORMAT INDIAN CURRENCY
# ==================================================

def format_inr(value):

    try:

        if pd.isna(value):

            return "—"

        return f"₹{float(value):,.0f}"

    except (ValueError, TypeError, OverflowError):

        return "—"


# ==================================================
# GET PRICE COLUMN
# ==================================================

def get_price_column(df):

    for column in [
        "selling_price",
        "price"
    ]:

        if column in df.columns:

            return column

    return None
