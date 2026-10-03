"""Predict systolic blood pressure from age and weight with linear regression.

Learning project: the dataset is a small, hand-made sample (15 rows),
so the results show the workflow, not a clinically useful model.
"""
import os

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Dataset (small synthetic sample)
data = {
    "age":    [45, 50, 55, 60, 65, 70, 35, 40, 48, 52, 58, 62, 67, 72, 75],
    "weight": [60, 65, 68, 72, 75, 78, 55, 58, 62, 66, 70, 73, 77, 80, 82],
    "bp":     [120, 122, 130, 135, 140, 145, 115, 118, 125, 128, 132, 138, 142, 148, 150],
}
df = pd.DataFrame(data)

X = df[["age", "weight"]]
y = df["bp"]

# 2. Train/test split (scaling happens inside the pipeline, fitted on training data only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# 3. Evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

# 5-fold cross-validation with shuffling, scored by mean absolute error
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

print("--- Model Performance ---")
print(f"R² (test set):          {r2:.3f}")
print(f"MAE (test set):         {mae:.2f} mmHg")
print(f"RMSE (test set):        {rmse:.2f} mmHg")
print(f"MAE (5-fold CV, mean):  {cv_mae.mean():.2f} mmHg")

# 4. Example prediction
example = pd.DataFrame({"age": [50], "weight": [70]})
print(f"\nPredicted BP for age 50, weight 70 kg: {model.predict(example)[0]:.1f} mmHg")

# 5. Charts
# charts are saved next to the script

plt.figure(figsize=(6, 4.5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)
plt.close()

all_preds = model.predict(X)
plt.figure(figsize=(6, 4.5))
plt.scatter(y, all_preds, color="#2b6cb0")
lims = [y.min() - 3, y.max() + 3]
plt.plot(lims, lims, "--", color="gray", label="Perfect prediction")
plt.xlabel("Actual BP (mmHg)")
plt.ylabel("Predicted BP (mmHg)")
plt.title("Predicted vs actual blood pressure")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.close()

print("\nCharts saved")
