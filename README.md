# Heart Disease Prediction System

A Streamlit web application that predicts heart disease risk using machine learning.

## Features
- Real-time heart disease risk prediction
- Interactive patient data input
- Machine learning model (Random Forest)
- User-friendly web interface

## Installation

```bash
pip install -r requirements.txt
Usage
streamlit run app.py

Project Structure
heart_disease_prediction/
├── app.py                 # Main Streamlit application
├── run_heart_project.py   # Application runner
├── setup.py              # Package setup
├── requirements.txt       # Python dependencies
└── models/               # Trained ML models
    ├── heart_model.pkl
    └── scaler.pkl

Model Details
Algorithm: Random Forest Classifier

Features: 13 clinical parameters

Purpose: Heart disease risk assessment
