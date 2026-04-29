import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/employee_risk_model.pkl")

st.set_page_config(page_title="Employee Risk Prediction System", layout="centered")

st.title("AI-Based Employee Performance Risk Prediction System")

st.write(
    "This application predicts whether an employee has Low, Medium or High performance risk based on HR-related data."
)

# User inputs
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, value=85)
late_days = st.number_input("Late Days", min_value=0, max_value=31, value=2)
absent_days = st.number_input("Absent Days", min_value=0, max_value=31, value=1)
tasks_completed = st.number_input("Tasks Completed", min_value=0, max_value=100, value=20)
average_task_rating = st.number_input("Average Task Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0)
overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=100, value=5)
previous_performance_score = st.number_input("Previous Performance Score", min_value=0, max_value=100, value=75)

# Predict button
if st.button("Predict Risk Level"):
    input_data = pd.DataFrame(
        [[
            attendance_rate,
            late_days,
            absent_days,
            tasks_completed,
            average_task_rating,
            overtime_hours,
            previous_performance_score,
        ]],
        columns=[
            "attendance_rate",
            "late_days",
            "absent_days",
            "tasks_completed",
            "average_task_rating",
            "overtime_hours",
            "previous_performance_score",
        ],
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("Low Risk: Employee performance is good.")
    elif prediction == "Medium":
        st.warning("Medium Risk: Employee may need some support.")
    else:
        st.error("High Risk: Employee may need attention or improvement support.")