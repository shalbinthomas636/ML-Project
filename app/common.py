
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_car_price_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

DATASET_PATHS = [
    ROOT / "outputs" / "cleaned_dataset.csv",
    ROOT / "dataset" / "cleaned_dataset.csv",
    ROOT / "dataset" / "Car details v3.csv",
    ROOT / "Car details v3.csv",
]

COMPARISON_PATHS = [
    ROOT / "outputs" / "model_comparison.csv",
    ROOT / "outputs" / "algorithm_comparison.csv",
    ROOT / "model_comparison.csv",
]


def find_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    path = find_existing(DATASET_PATHS)

    if path is None:
        raise FileNotFoundError(
            "Dataset not found. Check the dataset folder."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "brand" not in df.columns and "name" in df.columns:
        df["brand"] = (
            df["name"].fillna("Unknown")
            .astype(str).str.split().str[0]
        )

    if "selling_price" in df.columns:
        df["selling_price"] = pd.to_numeric(
            df["selling_price"], errors="coerce"
        )

    return df


@st.cache_data
def load_comparison():
    path = find_existing(COMPARISON_PATHS)

    if path is None:
        return pd.DataFrame()

    return pd.read_csv(path)


def get_feature_lists(metadata):
    numeric = metadata.get("numeric_features", [])
    categorical = metadata.get("categorical_features", [])
    features = metadata.get("feature_columns", numeric + categorical)

    return features, numeric, categorical


def build_prediction_input(values, metadata):
    features, numeric, categorical = get_feature_lists(metadata)

    row = {}

    for feature in features:
        if feature in values:
            row[feature] = values[feature]
        elif feature in metadata.get("numeric_defaults", {}):
            row[feature] = metadata["numeric_defaults"][feature]
        elif feature in metadata.get("category_options", {}):
            options = metadata["category_options"][feature]
            row[feature] = options[0] if options else "Unknown"
        else:
            row[feature] = 0

    return pd.DataFrame([row], columns=features)


def format_inr(value):
    return "₹" + format(float(value), ",.0f")


def get_price_column(df):
    for column in ["selling_price", "price"]:
        if column in df.columns:
            return column
    return None
