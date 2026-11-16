
import os
import re
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "Car details v3.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("=" * 60)
print("AUTOVAL - USED CAR PRICE PREDICTION")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

df = df.loc[
    :,
    ~df.columns.str.contains(r"^unnamed|^index$")
]

column_mapping = {
    "price": "selling_price",
    "sellingprice": "selling_price",
    "present_price": "selling_price",
    "kms_driven": "km_driven",
    "kilometers_driven": "km_driven",
    "car_name": "name",
    "manufacturing_year": "year"
}

df = df.rename(columns=column_mapping)

if "selling_price" not in df.columns:
    raise ValueError("selling_price column is missing.")

df = df.drop_duplicates().copy()

print("Original cleaned shape:", df.shape)

# --------------------------------------------------
# 2. NUMERIC CONVERSION
# --------------------------------------------------

def extract_number(value):

    if pd.isna(value):
        return np.nan

    value = str(value).replace(",", "").strip()

    match = re.search(r"-?\d+\.?\d*", value)

    return float(match.group()) if match else np.nan


numeric_columns = [
    "selling_price",
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = df[col].apply(extract_number)

# --------------------------------------------------
# 3. BRAND EXTRACTION
# --------------------------------------------------

if "name" in df.columns:

    df["brand"] = (
        df["name"]
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
    )

    df["brand"] = df["brand"].replace(
        ["nan", "None", ""],
        np.nan
    )

# Keep the original name column because it is useful
# for displaying and searching vehicles.
# It is not necessary to use it as a model feature.

# --------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

df = df[
    df["selling_price"].notna()
    & (df["selling_price"] > 0)
].copy()

if "year" in df.columns:

    df.loc[
        (df["year"] < 1950) |
        (df["year"] > 2026),
        "year"
    ] = np.nan

if "km_driven" in df.columns:

    df.loc[
        df["km_driven"] < 0,
        "km_driven"
    ] = np.nan

for col in ["mileage", "engine", "max_power", "seats"]:

    if col in df.columns:

        df.loc[df[col] <= 0, col] = np.nan

if "torque" in df.columns:
    df.drop(columns=["torque"], inplace=True)

if "name" in df.columns:
    df.drop(columns=["name"], inplace=True)

df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_car_data.csv"),
    index=False
)

print("Cleaned dataset:", df.shape)

# --------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    df["selling_price"],
    kde=True
)

plt.title("Used Car Price Distribution")
plt.xlabel("Selling Price (INR)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_distribution.png")
)

plt.close()

if "year" in df.columns:

    plt.figure(figsize=(10, 5))

    sns.scatterplot(
        data=df,
        x="year",
        y="selling_price"
    )

    plt.title("Year vs Selling Price")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "year_vs_price.png")
    )

    plt.close()

if "fuel" in df.columns:

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="fuel",
        y="selling_price"
    )

    plt.xticks(rotation=25)
    plt.title("Fuel Type vs Selling Price")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fuel_vs_price.png")
    )

    plt.close()

# --------------------------------------------------
# 6. FEATURES AND TARGET
# --------------------------------------------------

TARGET = "selling_price"

X = df.drop(columns=[TARGET]).copy()

y = df[TARGET].copy()

X = X.dropna(axis=1, how="all")

constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X.drop(columns=constant_columns, inplace=True)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("Numerical:", numeric_features)
print("Categorical:", categorical_features)

# --------------------------------------------------
# 7. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 8. PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                min_frequency=5
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    sparse_threshold=0
)

# --------------------------------------------------
# 9. ALGORITHMS
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=10),

    "Lasso Regression": Lasso(
        alpha=0.01,
        max_iter=10000
    ),

    "ElasticNet Regression": ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10
    )
}

# --------------------------------------------------
# 10. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
trained_models = {}

for name, algorithm in models.items():

    print("Training:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", algorithm)
        ]
    )

    try:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        mape = mean_absolute_percentage_error(
            y_test,
            predictions
        ) * 100

        results.append({
            "Algorithm": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "MAPE (%)": mape
        })

        trained_models[name] = pipeline

        print("RMSE:", round(rmse, 2))

    except Exception as error:

        print("Failed:", name, error)

if not results:
    raise RuntimeError("No model could be trained.")

# --------------------------------------------------
# 11. COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE",
    ascending=True
).reset_index(drop=True)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "algorithm_comparison.csv"),
    index=False
)

best_algorithm = results_df.iloc[0]["Algorithm"]

best_model = trained_models[best_algorithm]

print("BEST MODEL:", best_algorithm)

# --------------------------------------------------
# 12. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_car_price_model.pkl")
)

# --------------------------------------------------
# 13. SAVE METADATA
# --------------------------------------------------

metadata = {

    "feature_columns": X.columns.tolist(),

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "numeric_defaults": {
        col: float(X_train[col].median())
        for col in numeric_features
    },

    "category_options": {
        col: sorted(
            X_train[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for col in categorical_features
    },

    "best_algorithm": best_algorithm,

    "model_metrics": results_df.iloc[0].to_dict(),

    "training_records": int(len(X_train)),

    "testing_records": int(len(X_test)),

    "dataset_records": int(len(df))
}

with open(
    os.path.join(MODEL_DIR, "model_metadata.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

# --------------------------------------------------
# 14. SAVE TEST PREDICTIONS
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": best_predictions,

    "Absolute Error": np.abs(
        y_test.values - best_predictions
    )

}).to_csv(
    os.path.join(OUTPUT_DIR, "best_model_predictions.csv"),
    index=False
)

# --------------------------------------------------
# 15. SAVE MODEL CHARTS
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="RMSE"
)

plt.xticks(rotation=40, ha="right")
plt.title("Algorithm RMSE Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "algorithm_comparison.png")
)

plt.close()

plt.figure(figsize=(12, 6))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="R2 Score"
)

plt.xticks(rotation=40, ha="right")
plt.title("Algorithm R2 Score Comparison")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "r2_comparison.png")
)

plt.close()

print("\nTRAINING COMPLETED")
print("Best algorithm:", best_algorithm)
print("Dataset records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(results_df.to_string(index=False))
