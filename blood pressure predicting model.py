import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Expanded Dataset (Adding more noise and rows for realism)
data = {
    'age': [45, 50, 55, 60, 65, 70, 35, 40, 48, 52, 58, 62, 67, 72, 75],
    'weight': [60, 65, 68, 72, 75, 78, 55, 58, 62, 66, 70, 73, 77, 80, 82],
    'bp': [120, 122, 130, 135, 140, 145, 115, 118, 125, 128, 132, 138, 142, 148, 150]
}
df = pd.DataFrame(data)

# 2. Preprocessing
X = df[['age', 'weight']]
y = df['bp']

# Feature Scaling: Models perform better when data is centered around 0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split with 'random_state' for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model Training & Cross-Validation
model = LinearRegression()
model.fit(X_train, y_train)

# Check consistency using Cross-Validation (shuffles data 5 times)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

# 5. Advanced Evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"--- Model Performance ---")
print(f"R² Score: {r2:.4f}") # How much variance is explained (1.0 is perfect)
print(f"Mean Absolute Error: {mae:.2f} mmHg")
print(f"Avg Cross-Val Score: {cv_scores.mean():.4f}")

# 6. Visualization (Correlation Heatmap)
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Map")
plt.show()
