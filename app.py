# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/heart_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.title("❤️ Heart Disease Prediction System")
st.markdown("Assess your heart disease risk using machine learning")

# Sidebar for patient input
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 100, 50)
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)

st.sidebar.subheader("Other Factors")
cp = st.sidebar.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", 
                          "Non-anginal Pain", "Asymptomatic"])
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

# Convert inputs to model format
input_mapping = {
    'cp': {"Typical Angina": 0, "Atypical Angina": 1, 
           "Non-anginal Pain": 2, "Asymptomatic": 3},
    'exang': {"No": 0, "Yes": 1},
    'fbs': {"No": 0, "Yes": 1}
}

# Prediction button
if st.sidebar.button("🔍 Predict Heart Disease Risk", type="primary"):
    # Prepare patient data
    patient_data = [[
        age, 1,  # age, sex (1 for male)
        input_mapping['cp'][cp],
        trestbps, chol,
        input_mapping['fbs'][fbs],
        1,  # restecg
        thalach,
        input_mapping['exang'][exang],
        oldpeak,
        1,  # slope
        0,  # ca
        2   # thal
    ]]
    
    # Scale and predict
    patient_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    # Display results
    st.header("🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("🔴 HEART DISEASE DETECTED")
        else:
            st.success("🟢 NO HEART DISEASE")
    
    with col2:
        st.metric("Probability", f"{probability:.2%}")
    
    with col3:
        if probability > 0.7:
            risk_level = "HIGH RISK"
            color = "red"
        elif probability > 0.3:
            risk_level = "MODERATE RISK"
            color = "orange"
        else:
            risk_level = "LOW RISK"
            color = "green"
        st.metric("Risk Level", risk_level)
    
    # Risk interpretation
    st.subheader("📊 Risk Analysis")
    if risk_level == "LOW RISK":
        st.info("""
        **Maintain Healthy Lifestyle:**
        - Continue regular exercise
        - Balanced diet with fruits and vegetables
        - Regular health checkups
        """)
    elif risk_level == "MODERATE RISK":
        st.warning("""
        **Consult Healthcare Provider:**
        - Schedule a doctor visit
        - Monitor blood pressure regularly
        - Consider cholesterol screening
        - Maintain healthy weight
        """)
    else:
        st.error("""
        **Seek Medical Advice Soon:**
        - Consult a cardiologist
        - Regular heart health screenings
        - Lifestyle changes essential
        - Medication may be needed
        """)
    
    # Patient summary
    st.subheader("📋 Patient Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Blood Pressure:** {trestbps}")
        st.write(f"**Cholesterol:** {chol}")
        st.write(f"**Max Heart Rate:** {thalach}")
    
    with col2:
        st.write(f"**Chest Pain:** {cp}")
        st.write(f"**Exercise Angina:** {exang}")
        st.write(f"**ST Depression:** {oldpeak}")
        st.write(f"**High Blood Sugar:** {fbs}")

# Disclaimer
st.markdown("---")
st.warning("""
**⚠️ Medical Disclaimer:** 
This tool is for educational purposes only. It is not a substitute for professional 
medical advice, diagnosis, or treatment. Always consult qualified healthcare 
providers for medical concerns.
""")

st.success("🎉 **Project Status:** Model trained with 86% accuracy")