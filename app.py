"""Streamlit app: estimate blood pressure and hypertension risk.

Run locally:
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from bp_model.features import FEATURES, HYPERTENSION_DIA, HYPERTENSION_SYS, add_features

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"

st.set_page_config(page_title="Blood Pressure Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    reg = joblib.load(MODELS / "bp_regressor.joblib")
    clf = joblib.load(MODELS / "hypertension_classifier.joblib")
    metrics = json.loads((MODELS / "metrics.json").read_text())
    return reg, clf, metrics


reg, clf, metrics = load_models()
best_reg = next(r for r in metrics["regression"]["results"]
                if r["model"] == metrics["regression"]["best_model"])
best_clf = next(r for r in metrics["classification"]["results"]
                if r["model"] == metrics["classification"]["best_model"])

st.title("🩺 Blood Pressure Predictor")
st.caption(
    f"Gradient-boosted models trained on {metrics['dataset']['rows_clean']:,} patient records. "
    "Educational project, not medical advice."
)

tab_predict, tab_model = st.tabs(["Predict", "How the model works"])

with tab_predict:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Your details")
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 30, 65, 50)
        sex = c2.radio("Sex", ["Female", "Male"], horizontal=True)
        height = c1.number_input("Height (cm)", 120, 220, 168)
        weight = c2.number_input("Weight (kg)", 30, 200, 72)
        cholesterol = c1.selectbox("Cholesterol", [1, 2, 3],
                                   format_func=lambda v: ["Normal", "Above normal", "Well above normal"][v - 1])
        gluc = c2.selectbox("Glucose", [1, 2, 3],
                            format_func=lambda v: ["Normal", "Above normal", "Well above normal"][v - 1])
        smoke = c1.toggle("Smoker")
        alco = c2.toggle("Drinks alcohol")
        active = st.toggle("Physically active", value=True)

    person = add_features(pd.DataFrame([{
        "age_years": age, "is_male": int(sex == "Male"), "height": height, "weight": weight,
        "cholesterol": cholesterol, "gluc": gluc, "smoke": int(smoke),
        "alco": int(alco), "active": int(active),
    }]))[FEATURES]

    systolic = float(reg.predict(person)[0])
    risk = float(clf.predict_proba(person)[0, 1])
    base_rate = metrics["dataset"]["hypertension_rate"]

    with right:
        st.subheader("Estimate")
        m1, m2, m3 = st.columns(3)
        m1.metric("Systolic BP", f"{systolic:.0f} mmHg",
                  help=f"Typical error ±{best_reg['test_mae']:.0f} mmHg")
        m2.metric("Hypertension risk", f"{risk:.0%}",
                  delta=f"{(risk - base_rate) * 100:+.0f} pts vs average",
                  delta_color="inverse")
        m3.metric("BMI", f"{person['bmi'].iloc[0]:.1f}")

        level = "High" if risk >= 0.5 else "Moderate" if risk >= 0.3 else "Low"
        st.progress(min(risk, 1.0), text=f"{level} risk of stage 2 hypertension "
                                         f"(≥{HYPERTENSION_SYS}/{HYPERTENSION_DIA} mmHg)")

        st.markdown("##### What if?")
        scenarios = {"As entered": person}
        if weight - 5 >= 30:
            scenarios["Weigh 5 kg less"] = add_features(person.assign(weight=weight - 5))[FEATURES]
        if cholesterol > 1:
            scenarios["Normal cholesterol"] = person.assign(cholesterol=1)
        if gluc > 1:
            scenarios["Normal glucose"] = person.assign(gluc=1)
        if not active:
            scenarios["Physically active"] = person.assign(active=1)
        rows = [{"Scenario": k,
                 "Systolic (mmHg)": round(float(reg.predict(v)[0])),
                 "Risk": f"{clf.predict_proba(v)[0, 1]:.0%}"} for k, v in scenarios.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Shows how the model's estimate changes. These are patterns in the data, "
                   "not proof that a change would cause the difference.")

        st.info(
            "These are population-level estimates from a public dataset. Blood pressure varies a lot "
            "between people with the same profile, so the only way to know yours is to measure it."
        )

with tab_model:
    st.subheader("Model performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{metrics['dataset']['rows_clean']:,}")
    c2.metric("Systolic MAE", f"{best_reg['test_mae']:.1f} mmHg")
    c3.metric("Hypertension AUC", f"{best_clf['test_auc']:.3f}")
    c4.metric("Accuracy", f"{best_clf['test_accuracy']:.1%}")

    st.image(str(FIGS / "model_comparison.png"))
    c1, c2 = st.columns(2)
    c1.image(str(FIGS / "feature_importance.png"))
    c2.image(str(FIGS / "roc_curve.png"))
    st.image(str(FIGS / "eda_distribution.png"))
