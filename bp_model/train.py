"""Train, compare and save the blood pressure models.

Two tasks:
  1. Regression     - predict systolic blood pressure (mmHg)
  2. Classification - predict stage 2 hypertension (>= 140/90)

Run from the project root:
    python -m bp_model.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import clean, load_raw
from .features import FEATURE_LABELS, FEATURES, add_features, hypertension_label

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def regressors():
    return {
        "Baseline (mean)": DummyRegressor(),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=20, n_jobs=-1, random_state=SEED
        ),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=SEED
        ),
    }


def classifiers():
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=20, n_jobs=-1, random_state=SEED
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=SEED
        ),
    }


def compare_regressors(X_train, y_train, X_test, y_test):
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in regressors().items():
        cv_mae = -cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "cv_mae": cv_mae.mean(),
            "cv_mae_std": cv_mae.std(),
            "test_mae": mean_absolute_error(y_test, pred),
            "test_rmse": root_mean_squared_error(y_test, pred),
            "test_r2": r2_score(y_test, pred),
        })
        fitted[name] = model
        print(f"  {name:<22} CV MAE {cv_mae.mean():6.2f}   test R² {rows[-1]['test_r2']:.3f}")
    return pd.DataFrame(rows), fitted


def compare_classifiers(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in classifiers().items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "cv_auc": cv_auc.mean(),
            "cv_auc_std": cv_auc.std(),
            "test_auc": roc_auc_score(y_test, proba),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_f1": f1_score(y_test, pred, zero_division=0),
        })
        fitted[name] = model
        print(f"  {name:<22} CV AUC {cv_auc.mean():.3f}   test acc {rows[-1]['test_accuracy']:.3f}")
    return pd.DataFrame(rows), fitted


# ---------- figures ----------

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["ap_hi"], bins=range(75, 245, 10), ax=axes[0], color=BLUE)
    axes[0].axvline(140, color=ORANGE, ls="--", label="Hypertension (140)")
    axes[0].set(title="Systolic blood pressure", xlabel="mmHg", ylabel="Patients")
    axes[0].legend()

    bins = pd.cut(df["age_years"], [29, 40, 45, 50, 55, 60, 66])
    rate = df.groupby(bins, observed=True)["hypertension"].mean() * 100
    axes[1].bar([f"{int(i.left)+1}-{int(i.right)}" for i in rate.index], rate.values, color=BLUE)
    axes[1].set(title="Hypertension rate by age", xlabel="Age group", ylabel="% of patients")
    save(fig, "eda_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    cols = FEATURES + ["ap_hi", "ap_lo"]
    corr = df[cols].corr()
    labels = [FEATURE_LABELS.get(c, {"ap_hi": "Systolic BP", "ap_lo": "Diastolic BP"}.get(c)) for c in cols]
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 8}, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Feature correlation")
    save(fig, "correlation.png")


def plot_model_comparison(reg, clf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [GREY] + [BLUE] * (len(reg) - 1)
    axes[0].barh(reg["model"], reg["cv_mae"], xerr=reg["cv_mae_std"], color=colors)
    axes[0].set(title="Systolic BP: error (lower is better)", xlabel="Mean absolute error, mmHg")
    axes[0].invert_yaxis()

    colors = [GREY] + [ORANGE] * (len(clf) - 1)
    axes[1].barh(clf["model"], clf["cv_auc"], xerr=clf["cv_auc_std"], color=colors)
    axes[1].set(title="Hypertension: ROC AUC (higher is better)", xlabel="ROC AUC", xlim=(0.45, 0.8))
    axes[1].invert_yaxis()
    save(fig, "model_comparison.png")


def plot_predictions(model, X_test, y_test):
    pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.hexbin(y_test, pred, gridsize=35, cmap="Blues", mincnt=1, bins="log")
    lims = [90, 200]
    ax.plot(lims, lims, "--", color=ORANGE, label="Perfect prediction")
    ax.set(xlim=lims, ylim=(100, 170), xlabel="Actual systolic BP (mmHg)",
           ylabel="Predicted (mmHg)", title="Predicted vs actual (test set)")
    ax.legend(loc="upper left")
    save(fig, "predicted_vs_actual.png")


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, model), color in zip(models.items(), [GREY, "#38a169", "#805ad5", ORANGE]):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="black", lw=1)
    ax.set(title="Hypertension classifier: ROC curves",
           xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "roc_curve.png")


def plot_importance(model, X_test, y_test):
    sample = X_test.sample(min(4000, len(X_test)), random_state=SEED)
    result = permutation_importance(model, sample, y_test.loc[sample.index],
                                    scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=[FEATURE_LABELS[c] for c in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=ORANGE)
    ax.set(title="What drives hypertension risk?", xlabel="Drop in ROC AUC when feature is shuffled")
    save(fig, "feature_importance.png")
    return imp.sort_values(ascending=False)


# ---------- main ----------

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, report = clean(load_raw())
    print(report.summary())
    df = add_features(df)
    df["hypertension"] = hypertension_label(df)

    X = df[FEATURES]
    y_reg, y_clf = df["ap_hi"], df["hypertension"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
    )

    print("\nRegression: systolic BP")
    reg_table, reg_models = compare_regressors(X_train, yr_train, X_test, yr_test)
    print("\nClassification: hypertension")
    clf_table, clf_models = compare_classifiers(X_train, yc_train, X_test, yc_test)

    best_reg = reg_table.iloc[1:].sort_values("cv_mae").iloc[0]["model"]
    best_clf = clf_table.iloc[1:].sort_values("cv_auc", ascending=False).iloc[0]["model"]
    print(f"\nBest regressor: {best_reg}   Best classifier: {best_clf}")

    plot_eda(df)
    plot_model_comparison(reg_table, clf_table)
    plot_predictions(reg_models[best_reg], X_test, yr_test)
    plot_roc(clf_models, X_test, yc_test)
    importance = plot_importance(clf_models[best_clf], X_test, yc_test)

    joblib.dump(reg_models[best_reg], MODELS_DIR / "bp_regressor.joblib", compress=3)
    joblib.dump(clf_models[best_clf], MODELS_DIR / "hypertension_classifier.joblib", compress=3)

    metrics = {
        "dataset": {
            "rows_raw": report.start_rows,
            "rows_clean": report.end_rows,
            "removed": report.removed,
            "hypertension_rate": round(float(y_clf.mean()), 4),
        },
        "regression": {"best_model": best_reg, "results": reg_table.round(4).to_dict("records")},
        "classification": {"best_model": best_clf, "results": clf_table.round(4).to_dict("records")},
        "feature_importance": importance.round(4).to_dict(),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models, metrics and figures.")


if __name__ == "__main__":
    main()
