import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/employee_data.csv")

# Input columns
X = data[
    [
        "attendance_rate",
        "late_days",
        "absent_days",
        "tasks_completed",
        "average_task_rating",
        "overtime_hours",
        "previous_performance_score",
    ]
]

# Output column
y = data["risk_level"]

# Split data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create machine learning model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Print results
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create model folder if it does not exist
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/employee_risk_model.pkl")

print("\nModel saved successfully!")