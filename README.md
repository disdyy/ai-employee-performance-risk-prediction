This project predicts employee performance risk levels using machine learning.

## Project Overview

The system uses HR-related data such as attendance rate, absent days, late days, task completion count, average task rating, overtime hours, and previous performance score to predict whether an employee is at Low, Medium, or High risk.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Features

- Predicts employee risk level
- Uses a trained machine learning model
- Provides a simple web interface
- Displays Low, Medium, or High risk prediction
- Beginner-friendly AI/ML project

## Dataset Features

- Attendance Rate
- Late Days
- Absent Days
- Tasks Completed
- Average Task Rating
- Overtime Hours
- Previous Performance Score

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
