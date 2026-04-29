# AI-Based Employee Performance Risk Prediction System

A machine learning project that predicts employee performance risk levels using HR-related data.

## Project Overview

This project predicts whether an employee is in a Low, Medium, or High risk category based on attendance, absenteeism, task completion, task rating, overtime hours and previous performance score.

The project was created to understand how machine learning can be applied in an HR and product-based environment.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Features

- Predicts employee performance risk level
- Uses HR-related employee data
- Trains a machine learning classification model
- Saves the trained model using Joblib
- Provides a simple Streamlit web interface
- Displays Low, Medium, or High risk prediction

## Dataset Features

The model uses the following input features:

- Attendance Rate
- Late Days
- Absent Days
- Tasks Completed
- Average Task Rating
- Overtime Hours
- Previous Performance Score

## Machine Learning Workflow

1. Load the employee dataset
2. Separate input features and target label
3. Split data into training and testing sets
4. Train a Random Forest Classifier
5. Evaluate the model
6. Save the trained model
7. Use the saved model in a Streamlit app

## How to Run the Project

Install required packages:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
