"""Feature engineering shared by training and the web app."""
from __future__ import annotations

import pandas as pd

FEATURES = [
    "age_years",
    "is_male",
    "height",
    "weight",
    "bmi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

FEATURE_LABELS = {
    "age_years": "Age",
    "is_male": "Sex (male)",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "cholesterol": "Cholesterol level",
    "gluc": "Glucose level",
    "smoke": "Smoker",
    "alco": "Drinks alcohol",
    "active": "Physically active",
}

# Stage 2 hypertension threshold (ACC/AHA): systolic >= 140 or diastolic >= 90
HYPERTENSION_SYS = 140
HYPERTENSION_DIA = 90


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns (currently BMI)."""
    df = df.copy()
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    return df


def hypertension_label(df: pd.DataFrame) -> pd.Series:
    """1 if the reading meets the stage 2 hypertension threshold."""
    return ((df["ap_hi"] >= HYPERTENSION_SYS) | (df["ap_lo"] >= HYPERTENSION_DIA)).astype(int)
