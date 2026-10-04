import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.xlsx")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model_pipeline_v2.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics_v2.json")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema_v2.json")

RANDOM_STATE = 42


# ============================================================
# RAW COLUMN NAMES
# ============================================================

AGE = "2. Age:"
YEAR = "3.Year of study:"
PLATFORMS = "4.Which social media platforms do you use regularly? (Select all that apply)"
DAILY_TIME = "5.Average daily time spent on social media:"
PURPOSE = "6.Purpose of social media use :"
TIMING = "7.When do you mostly use social media"
MULTITASKING = "8.Do you engage in social media multitasking while studying or completing assignments?"
STUDY_HOURS = "9.Do you use social media during study hours?"
STUDY_FREQUENCY = "10.On a scale of 1 to 10, how often do you use social media during study time"
CONCENTRATION = "11.Effect of social media on your concentration"
ACADEMIC_IMPACT = "12.On a scale of 1 to 10, how much social media has affected your academic performances/grades"
ACADEMIC_USE = "13.Do you use social media for academic purposes (e.g., study groups, resource sharing, consultations with professors)?"
MISSED_DEADLINES = "14.Has social media access ever led to missed deadlines or incomplete assignments?"

TARGET = "CGPA"


# ============================================================
# PLATFORM CLEANING
# ============================================================

PLATFORM_MAP = {
    "Instagram": "Instagram",
    "WhatsApp": "WhatsApp",
    "YouTube": "YouTube",
    "Snapchat": "Snapchat",
    "Reddit": "Reddit",
    "LinkedIn": "LinkedIn",
    "Linkdin": "LinkedIn",
    "LinkdIn": "LinkedIn",
    "Discord": "Discord",
    "pintrest": "Pinterest",
    "Pinterest": "Pinterest",
    "grok": "Grok",
    "AI fiesta": "AI Fiesta",
}

PLATFORMS_ALLOWED = [
    "Instagram",
    "WhatsApp",
    "YouTube",
    "Snapchat",
    "Reddit",
    "LinkedIn",
    "Discord",
    "Pinterest",
    "Grok",
    "AI Fiesta",
]


def clean_platforms(value):
    if pd.isna(value):
        return ""

    parts = [x.strip() for x in str(value).split(",")]
    cleaned = []

    for part in parts:
        if not part:
            continue

        if part == "8":
            continue

        canonical = PLATFORM_MAP.get(part)

        if canonical is not None:
            cleaned.append(canonical)

    return ",".join(dict.fromkeys(cleaned))


def platform_flags(series):
    output = pd.DataFrame(index=series.index)

    for platform in PLATFORMS_ALLOWED:
        output[f"platform_{platform}"] = series.apply(
            lambda x: int(platform in str(x).split(","))
        )

    output["platform_count"] = output.sum(axis=1)

    return output


# ============================================================
# PURPOSE CLEANING
# ============================================================

PURPOSE_ALLOWED = [
    "Academic learning",
    "Entertainment",
    "Communication with friends/family",
    "Professional networking",
]


def purpose_flags(series):
    output = pd.DataFrame(index=series.index)

    for purpose in PURPOSE_ALLOWED:
        output[f"purpose_{purpose}"] = series.apply(
            lambda x: int(purpose in str(x))
        )

    output["purpose_count"] = output.sum(axis=1)

    return output


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def load_and_prepare():
    df = pd.read_excel(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Normalize yes/no
    df[STUDY_HOURS] = (
        df[STUDY_HOURS]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": "yes",
            "no": "no",
            "nan": np.nan,
        })
    )

    # Numeric conversion
    df[AGE] = pd.to_numeric(df[AGE], errors="coerce")
    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce")
    df[DAILY_TIME] = pd.to_numeric(df[DAILY_TIME], errors="coerce")
    df[STUDY_FREQUENCY] = pd.to_numeric(df[STUDY_FREQUENCY], errors="coerce")
    df[ACADEMIC_IMPACT] = pd.to_numeric(df[ACADEMIC_IMPACT], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Obvious data-entry errors
    df.loc[(df[AGE] < 15) | (df[AGE] > 25), AGE] = np.nan
    df.loc[(df[DAILY_TIME] < 0) | (df[DAILY_TIME] > 24), DAILY_TIME] = np.nan
    df.loc[
        (df[STUDY_FREQUENCY] < 0) | (df[STUDY_FREQUENCY] > 10),
        STUDY_FREQUENCY,
    ] = np.nan
    df.loc[
        (df[ACADEMIC_IMPACT] < 0) | (df[ACADEMIC_IMPACT] > 10),
        ACADEMIC_IMPACT,
    ] = np.nan

    # Keep only plausible CGPA targets
    df = df[df[TARGET].between(2.0, 10.0)].copy()

    # Clean multi-label fields
    df["_platforms_clean"] = df[PLATFORMS].apply(clean_platforms)

    platform_df = platform_flags(df["_platforms_clean"])
    purpose_df = purpose_flags(df[PURPOSE])

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    X = pd.DataFrame(index=df.index)

    X["age"] = df[AGE]
    X["year_of_study"] = df[YEAR]
    X["daily_time_hours"] = df[DAILY_TIME]
    X["study_time_frequency"] = df[STUDY_FREQUENCY]

    X["usage_timing"] = df[TIMING]
    X["multitasking"] = df[MULTITASKING]
    X["use_during_study_hours"] = df[STUDY_HOURS]
    X["concentration_effect"] = df[CONCENTRATION]
    X["academic_purpose_use"] = df[ACADEMIC_USE]
    X["missed_deadlines"] = df[MISSED_DEADLINES]

    X = pd.concat(
        [
            X,
            platform_df,
            purpose_df,
        ],
        axis=1,
    )

    y = df[TARGET]

    return df, X, y


# ============================================================
# FEATURE GROUPS
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "year_of_study",
    "daily_time_hours",
    "study_time_frequency",
    "platform_count",
    "purpose_count",
]

CATEGORICAL_FEATURES = [
    "usage_timing",
    "multitasking",
    "use_during_study_hours",
    "concentration_effect",
    "academic_purpose_use",
    "missed_deadlines",
]

PLATFORM_FEATURES = [
    f"platform_{platform}"
    for platform in PLATFORMS_ALLOWED
]

PURPOSE_FEATURES = [
    f"purpose_{purpose}"
    for purpose in PURPOSE_ALLOWED
]

NUMERIC_FEATURES += PLATFORM_FEATURES + PURPOSE_FEATURES


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=2.0)),
        ]
    )


# ============================================================
# FEATURE SCHEMA
# ============================================================

def build_feature_schema(df):
    schema = {
        "numeric_ranges": {},
        "categorical_options": {},
        "multi_label_options": {},
    }

    for col in [
        "age",
        "year_of_study",
        "daily_time_hours",
        "study_time_frequency",
    ]:
        source_col = {
            "age": AGE,
            "year_of_study": YEAR,
            "daily_time_hours": DAILY_TIME,
            "study_time_frequency": STUDY_FREQUENCY,
        }[col]

        schema["numeric_ranges"][col] = {
            "min": float(df[source_col].min(skipna=True)),
            "max": float(df[source_col].max(skipna=True)),
            "median": float(df[source_col].median(skipna=True)),
        }

    for col in CATEGORICAL_FEATURES:
        raw_col = {
            "usage_timing": TIMING,
            "multitasking": MULTITASKING,
            "use_during_study_hours": STUDY_HOURS,
            "concentration_effect": CONCENTRATION,
            "academic_purpose_use": ACADEMIC_USE,
            "missed_deadlines": MISSED_DEADLINES,
        }[col]

        values = (
            df[raw_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        schema["categorical_options"][col] = sorted(values)

    # IMPORTANT:
    # These are the canonical values used by the V2 training code.
    # Do not expose raw misspellings such as "pintrest", "grok",
    # "AI fiesta", or the garbage value "8".
    schema["multi_label_options"]["platforms"] = PLATFORMS_ALLOWED

    schema["multi_label_options"]["purpose"] = PURPOSE_ALLOWED

    return schema


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("TRAINING MODEL V2")
    print("=" * 70)

    df, X, y = load_and_prepare()

    print(f"Cleaned training rows: {len(X)}")
    print(f"Feature count before preprocessing: {X.shape[1]}")

    # --------------------------------------------------------
    # Hold-out evaluation
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    evaluation_model = build_pipeline()
    evaluation_model.fit(X_train, y_train)

    y_pred = evaluation_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # --------------------------------------------------------
    # 5-fold cross-validation
    # --------------------------------------------------------

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )

    cv_mae = -cv_scores["test_mae"]
    cv_rmse = -cv_scores["test_rmse"]
    cv_r2 = cv_scores["test_r2"]

    metrics = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2": round(float(
            evaluation_model.score(X_test, y_test)
        ), 4),
        "cv_r2_mean": round(float(cv_r2.mean()), 4),
        "cv_r2_std": round(float(cv_r2.std()), 4),
        "cv_mae_mean": round(float(cv_mae.mean()), 4),
        "cv_mae_std": round(float(cv_mae.std()), 4),
        "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
        "cv_rmse_std": round(float(cv_rmse.std()), 4),
        "random_state": RANDOM_STATE,
        "model": "Ridge",
        "ridge_alpha": 2.0,
        "features_used": (
            [
                "age",
                "year_of_study",
                "daily_time_hours",
                "study_time_frequency",
                "usage_timing",
                "multitasking",
                "use_during_study_hours",
                "concentration_effect",
                "academic_purpose_use",
                "missed_deadlines",
                "platforms",
                "purpose",
            ]
        ),
        "target": "cgpa",
        "notes": (
            "Personal identifiers, CGPA_Binary, the compared-to-previous-"
            "semester field, and the self-reported academic-impact score "
            "were excluded. Platform spellings and obvious garbage values "
            "were normalized before feature construction."
        ),
    }

    # --------------------------------------------------------
    # Final model trained on all available data
    # --------------------------------------------------------

    final_model = build_pipeline()
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_PATH)

    # --------------------------------------------------------
    # Save V2 metrics/schema separately
    # --------------------------------------------------------

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    schema = build_feature_schema(df)

    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    print(f"MAE :  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"RMSE:  {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"R²  :  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(MODEL_PATH)
    print(METRICS_PATH)
    print(SCHEMA_PATH)

    print("\nDONE.")


if __name__ == "__main__":
    main()
