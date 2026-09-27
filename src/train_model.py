"""
train_model.py
---------------
Trains the corrected Linear Regression model for CGPA prediction.

Run this script from the project root:

    python src/train_model.py

It will:
  1. Load and clean the dataset (data/dataset.xlsx).
  2. Build a preprocessing + Linear Regression pipeline.
  3. Evaluate it with a train/test split AND 5-fold cross-validation.
  4. Save the fitted pipeline to model/model_pipeline.joblib.
  5. Save the honest evaluation metrics to model/model_metrics.json.
  6. Save the feature schema (allowed categorical values, numeric
     ranges) to model/feature_schema.json so the Streamlit app can
     build its input form without hard-coding choices twice.

No metrics are invented anywhere in this file. Whatever this script
prints and saves is exactly what the trained model produced on this
run of this dataset.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(__file__))
from preprocessing import (
    CATEGORICAL_COLUMNS,
    MULTI_LABEL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
    MultiLabelOneHotEncoder,
    load_and_prepare,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.xlsx")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Build the preprocessing + Linear Regression pipeline.

    Numeric columns  -> median impute + standard scale
    Categorical cols -> most-frequent impute + one-hot encode
    Multi-select cols -> custom multi-label one-hot encode (handled
                          via a FeatureUnion so each multi-select
                          column gets its own encoder instance)
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
        + [
            (f"multilabel_{col}", MultiLabelOneHotEncoder(column=col), [col])
            for col in MULTI_LABEL_COLUMNS
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", column_transformer),
            ("model", LinearRegression()),
        ]
    )
    return pipeline


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = r2_score(y_test, y_pred)

    # 5-fold cross-validation on the FULL dataset for a more reliable
    # estimate than a single train/test split.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    cv_mae_scores = -cross_val_score(
        pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error"
    )

    metrics = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2": round(float(test_r2), 4),
        "cv_r2_mean": round(float(cv_r2_scores.mean()), 4),
        "cv_r2_std": round(float(cv_r2_scores.std()), 4),
        "cv_mae_mean": round(float(cv_mae_scores.mean()), 4),
        "cv_mae_std": round(float(cv_mae_scores.std()), 4),
        "random_state": RANDOM_STATE,
        "features_used": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + MULTI_LABEL_COLUMNS,
        "target": TARGET,
        "notes": (
            "Personal identifiers, CGPA_Binary, the 'compared to previous "
            "semester' field, and the self-reported 1-10 academic-impact "
            "score were excluded from the feature set to avoid data "
            "leakage. This is the corrected model; it replaces an earlier "
            "invalid run that accidentally included leaking features."
        ),
    }
    return metrics


def build_feature_schema(df: pd.DataFrame) -> dict:
    """Record the allowed values / ranges seen in the training data so
    the Streamlit app can build dropdowns that always match what the
    model was actually trained on.
    """
    schema = {"numeric_ranges": {}, "categorical_options": {}, "multi_label_options": {}}

    for col in NUMERIC_COLUMNS:
        schema["numeric_ranges"][col] = {
            "min": float(df[col].min(skipna=True)),
            "max": float(df[col].max(skipna=True)),
            "median": float(df[col].median(skipna=True)),
        }

    for col in CATEGORICAL_COLUMNS:
        schema["categorical_options"][col] = sorted(
            v for v in df[col].dropna().unique().tolist()
        )

    for col in MULTI_LABEL_COLUMNS:
        labels = set()
        for value in df[col].dropna():
            for item in str(value).split(","):
                item = item.strip()
                if item:
                    labels.add(item)
        schema["multi_label_options"][col] = sorted(labels)

    return schema


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_prepare(DATA_PATH)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + MULTI_LABEL_COLUMNS]
    y = df[TARGET]

    pipeline = build_pipeline()
    metrics = evaluate(pipeline, X, y)

    # Refit on the FULL dataset for the version we actually ship, now
    # that evaluation is already done on the held-out split above.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    schema = build_feature_schema(df)
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to:   {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Schema saved to:  {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
