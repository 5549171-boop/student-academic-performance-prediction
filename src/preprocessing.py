"""
preprocessing.py
-----------------
Data loading and cleaning for the Student Academic Performance
Prediction & Social Media Impact Analysis System.

This module is deliberately kept simple and well-commented so it can be
explained during a viva. It implements exactly the cleaning rules and
exclusion rules described in the project handoff document:

  - Personal identifiers (Timestamp, Email, Roll No, Name) are dropped.
  - CGPA is the target and is never used as a predictor.
  - CGPA_Binary is excluded (reserved for a future phase).
  - The "compared to previous semester" field is excluded because it is
    itself an academic-outcome question and would leak information
    about the target.
  - The "how much social media has affected your academic
    performance/grades (1-10)" field is also excluded. It is a
    self-reported judgement about impact on grades, which overlaps
    conceptually with the target (CGPA) rather than being an
    independent behavioural measurement, so keeping it would risk the
    same kind of leakage as the "compared to previous semester" field.
  - Invalid numeric responses are converted to missing values so they
    can be imputed rather than silently trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------
# Column name mapping (raw Google Form column -> short internal name)
# ---------------------------------------------------------------------
RAW_COLUMNS = {
    "timestamp": "Timestamp",
    "email": "Email Address",
    "roll_no": "Roll No (Student Collecting Data)",
    "name": "1.Name:",
    "age": "2. Age:",
    "year_of_study": "3.Year of study:",
    "platforms": "4.Which social media platforms do you use regularly? (Select all that apply)",
    "daily_time_hours": "5.Average daily time spent on social media:",
    "purpose": "6.Purpose of social media use :",
    "usage_timing": "7.When do you mostly use social media",
    "multitasking": "8.Do you engage in social media multitasking while studying or completing assignments?",
    "use_during_study_hours": "9.Do you use social media during study hours?",
    "study_time_frequency": "10.On a scale of 1 to 10, how often do you use social media during study time",
    "concentration_effect": "11.Effect of social media on your concentration",
    "self_reported_academic_impact": "12.On a scale of 1 to 10, how much social media has affected your academic performances/grades",
    "academic_purpose_use": "13.Do you use social media for academic purposes (e.g., study groups, resource sharing, consultations with professors)?",
    "missed_deadlines": "14.Has social media access ever led to missed deadlines or incomplete assignments?",
    "cgpa": "CGPA",
    "performance_vs_previous": "16.Compared to previous semester or exams, your current academic performance is:",
    "cgpa_binary": "CGPA_Binary",
}

TARGET = "cgpa"

# Columns that are personal identifiers -> always dropped, never features.
IDENTIFIER_COLUMNS = ["timestamp", "email", "roll_no", "name"]

# Columns excluded to avoid leakage / because they are reserved for a
# later phase of the project. See module docstring for the reasoning.
LEAKAGE_OR_RESERVED_COLUMNS = [
    "performance_vs_previous",   # academic outcome, leaks into CGPA
    "self_reported_academic_impact",  # self-reported grade impact, leaks into CGPA
    "cgpa_binary",                # reserved for future classification phase
]

# Multi-select (comma separated) categorical columns.
MULTI_LABEL_COLUMNS = ["platforms", "purpose"]

# The platforms question allowed free text, which introduced a few
# spelling variants of the same platform (e.g. "Linkdin" vs "LinkedIn").
# This mapping folds obvious variants together before encoding so they
# don't each get their own separate (and mostly empty) dummy column.
# Anything not listed here is left as-is; genuinely novel entries are
# safely ignored at prediction time by MultiLabelOneHotEncoder.
PLATFORM_ALIASES = {
    "linkdin": "LinkedIn",
    "linkdln": "LinkedIn",
}

# Single-choice categorical columns (one-hot encoded).
CATEGORICAL_COLUMNS = [
    "usage_timing",
    "multitasking",
    "use_during_study_hours",
    "concentration_effect",
    "academic_purpose_use",
    "missed_deadlines",
]

# Numeric columns used as model inputs.
NUMERIC_COLUMNS = [
    "age",
    "year_of_study",
    "daily_time_hours",
    "study_time_frequency",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + MULTI_LABEL_COLUMNS


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the raw Excel export and rename columns to short internal names."""
    df = pd.read_excel(path)
    inverse_map = {v: k for k, v in RAW_COLUMNS.items()}
    df = df.rename(columns=inverse_map)
    return df


def _clean_age(series: pd.Series) -> pd.Series:
    """Ages outside a plausible undergraduate range are treated as missing.

    The raw data contains values from 10 to 44. A B.Sc. student is
    realistically between roughly 15 and 30 years old; values outside
    that window are very likely data-entry errors and are set to NaN so
    they get median-imputed instead of distorting the model.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= 15) & (s <= 30))


def _clean_daily_time(series: pd.Series) -> pd.Series:
    """Daily social-media time must be between 0 and 24 hours."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s > 0) & (s <= 24))


def _clean_study_time_frequency(series: pd.Series) -> pd.Series:
    """Study-time frequency is on a 0-10 scale; anything outside that is invalid."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= 0) & (s <= 10))


def _normalize_platforms(value) -> str:
    """Fold known spelling variants of the same platform together."""
    if pd.isna(value):
        return value
    items = [i.strip() for i in str(value).split(",") if i.strip()]
    normalized = [PLATFORM_ALIASES.get(i.lower(), i) for i in items]
    return ", ".join(normalized)


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    """Normalize free-form yes/no style answers to a consistent casing."""
    return series.astype(str).str.strip().str.lower().replace({"nan": np.nan})


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a dataframe ready for feature
    selection. Does NOT drop identifier / leakage columns yet -- that
    happens in `select_model_columns` so the raw cleaned frame can also
    be used for exploratory analysis if needed.
    """
    df = df.copy()

    df["age"] = _clean_age(df["age"])
    df["daily_time_hours"] = _clean_daily_time(df["daily_time_hours"])
    df["study_time_frequency"] = _clean_study_time_frequency(df["study_time_frequency"])

    for col in ["use_during_study_hours", "academic_purpose_use"]:
        df[col] = _normalize_yes_no(df[col])

    df["platforms"] = df["platforms"].apply(_normalize_platforms)

    # year_of_study, multitasking, concentration_effect, missed_deadlines,
    # usage_timing are already clean categorical text in this dataset.

    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe containing only the target + approved feature
    columns, with identifiers and leakage/reserved columns dropped.
    """
    keep = FEATURE_COLUMNS + [TARGET]
    return df[keep].copy()


def load_and_prepare(path: str) -> pd.DataFrame:
    """Convenience wrapper: load, clean, and select columns in one call."""
    raw = load_raw_dataset(path)
    cleaned = clean_dataset(raw)
    return select_model_columns(cleaned)


# ---------------------------------------------------------------------
# Custom transformer for multi-select ("select all that apply") fields
# ---------------------------------------------------------------------
class MultiLabelOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes a comma-separated multi-select column.

    Example: "Instagram, WhatsApp, YouTube" -> separate 0/1 columns for
    Instagram, WhatsApp, YouTube, etc.

    Labels seen during `fit` define the output columns. Any label seen
    only at `transform` time (e.g. a brand-new platform typed by a
    future user) is safely ignored instead of crashing the app.
    """

    def __init__(self, column: str, sep: str = ","):
        self.column = column
        self.sep = sep

    def fit(self, X: pd.DataFrame, y=None):
        labels = set()
        for value in X[self.column].fillna(""):
            for item in str(value).split(self.sep):
                item = item.strip()
                if item:
                    labels.add(item)
        self.labels_ = sorted(labels)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = []
        for value in X[self.column].fillna(""):
            items = {i.strip() for i in str(value).split(self.sep) if i.strip()}
            rows.append([1 if lbl in items else 0 for lbl in self.labels_])
        return np.array(rows, dtype=int)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.column}__{lbl}" for lbl in self.labels_])
