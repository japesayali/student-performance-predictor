import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("student.csv")

print(data.head())

# Input features
X = data[["study_hours", "attendance", "sleep_hours"]]

# Output target
y = data["marks"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Check error
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Save model
joblib.dump(model, "model/student_model.pkl")

print("Model Saved Successfully!")