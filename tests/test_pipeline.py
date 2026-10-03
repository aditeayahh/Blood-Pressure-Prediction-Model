"""Tests for cleaning, features and the saved models.

Uses a tiny hand-made dataset so the tests run without downloading the real data.
"""
from pathlib import Path

import joblib
import pandas as pd
import pytest

from bp_model.data import clean
from bp_model.features import FEATURES, add_features, hypertension_label

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw():
    return pd.DataFrame({
        "id":          [1, 2, 3, 4, 5, 6],
        "age":         [18393, 20228, 18857, 17623, 17474, 18393],  # days
        "gender":      [2, 1, 1, 2, 1, 2],
        "height":      [168, 156, 165, 169, 50, 168],    # row 5: impossible height
        "weight":      [62.0, 85.0, 64.0, 82.0, 56.0, 62.0],
        "ap_hi":       [110, 140, 130, 150, 100, 110],
        "ap_lo":       [80, 90, 70, 100, 60, 80],
        "cholesterol": [1, 3, 3, 1, 1, 1],
        "gluc":        [1, 1, 1, 1, 1, 1],
        "smoke":       [0, 0, 0, 0, 0, 0],
        "alco":        [0, 0, 0, 0, 0, 0],
        "active":      [1, 1, 0, 1, 0, 1],
        "cardio":      [0, 1, 1, 1, 0, 0],
    })  # row 6 duplicates row 1


def test_clean_removes_bad_and_duplicate_rows(raw):
    df, report = clean(raw)
    assert len(df) == 4
    assert report.removed["duplicate rows"] == 1
    assert report.removed["height outside 120-220"] == 1
    assert report.end_rows == len(df)


def test_clean_rejects_diastolic_above_systolic(raw):
    raw.loc[0, ["ap_hi", "ap_lo"]] = [90, 120]
    df, report = clean(raw)
    assert report.removed["diastolic >= systolic"] == 1
    assert (df["ap_lo"] < df["ap_hi"]).all()


def test_clean_converts_age_and_gender(raw):
    df, _ = clean(raw)
    assert df["age_years"].between(40, 60).all()
    assert set(df["is_male"]) <= {0, 1}
    assert "age" not in df and "gender" not in df


def test_bmi():
    df = add_features(pd.DataFrame({"height": [180], "weight": [81]}))
    assert df["bmi"].iloc[0] == pytest.approx(25.0)


@pytest.mark.parametrize("sys_bp, dia_bp, expected", [
    (120, 80, 0), (139, 89, 0), (140, 80, 1), (130, 90, 1),
])
def test_hypertension_threshold(sys_bp, dia_bp, expected):
    df = pd.DataFrame({"ap_hi": [sys_bp], "ap_lo": [dia_bp]})
    assert hypertension_label(df).iloc[0] == expected


@pytest.mark.skipif(not (ROOT / "models" / "bp_regressor.joblib").exists(), reason="models not trained")
def test_saved_models_make_sensible_predictions(raw):
    reg = joblib.load(ROOT / "models" / "bp_regressor.joblib")
    clf = joblib.load(ROOT / "models" / "hypertension_classifier.joblib")
    df, _ = clean(raw)
    X = add_features(df)[FEATURES]

    systolic = reg.predict(X)
    risk = clf.predict_proba(X)[:, 1]
    assert ((systolic > 90) & (systolic < 200)).all()
    assert ((risk >= 0) & (risk <= 1)).all()


@pytest.mark.skipif(not (ROOT / "models" / "hypertension_classifier.joblib").exists(),
                    reason="models not trained")
def test_risk_increases_with_age_and_weight():
    clf = joblib.load(ROOT / "models" / "hypertension_classifier.joblib")
    base = {"is_male": 1, "height": 170, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1}
    young_light = add_features(pd.DataFrame([{**base, "age_years": 35, "weight": 65}]))[FEATURES]
    old_heavy = add_features(pd.DataFrame([{**base, "age_years": 62, "weight": 105}]))[FEATURES]
    assert clf.predict_proba(old_heavy)[0, 1] > clf.predict_proba(young_light)[0, 1]
