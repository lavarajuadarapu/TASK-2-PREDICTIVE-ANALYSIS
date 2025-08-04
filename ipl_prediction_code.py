
# IPL Runs Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("ipl_prediction_data.csv")

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df[['Team', 'Batsman', 'Bowler']])

# Define features and target
X = pd.concat([df[['Balls Faced', 'Fours', 'Sixes']], df_encoded], axis=1)
y = df['Runs Scored']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score:", r2)

# Save output to file
with open("ipl_output.txt", "w") as f:
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
    f.write(f"R-squared Score: {r2:.2f}\n")

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.bar(range(len(y_test)), y_test, width=0.4, label='Actual', align='edge')
plt.bar(range(len(y_pred)), y_pred, width=-0.4, label='Predicted', align='edge')
plt.xlabel("Sample")
plt.ylabel("Runs Scored")
plt.title("Actual vs Predicted Runs")
plt.legend()
plt.tight_layout()
plt.savefig("ipl_prediction_chart.png")
plt.show()
