# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 📊 Step 2: Create a small dataset (Example)
# This is fake data for learning
data = {
    'age': [45, 50, 55, 60, 65, 70],
    'weight': [60, 65, 68, 72, 75, 78],
    'bp': [120, 122, 130, 135, 140, 145]
}
df = pd.DataFrame(data)

# ✅ Step 3: Split into features (X) and target (y)
X = df[['age', 'weight']]
y = df['bp']

# 🔀 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 🤖 Step 5: Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Step 6: Predict and Show Result
preds = model.predict(X_test)
print("Predicted BP:", preds)
print("Actual BP:   ", y_test.values)

# 🔍 Step 7: Visualize (Just Age vs BP for simplicity)
plt.scatter(df['age'], df['bp'], color='blue')
plt.plot(df['age'], model.predict(df[['age', 'weight']]), color='red')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.title('Age vs Blood Pressure')
plt.show()
