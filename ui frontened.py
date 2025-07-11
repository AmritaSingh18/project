import streamlit as st
import joblib
import numpy as np

st.title("🎓 Student Performance Predictor")

# Load model
model = joblib.load('student_model.pkl')

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 15, 22, 17)
study_time = st.selectbox("Study Time (hrs/week)", [1, 2, 3, 4])
failures = st.slider("Past Class Failures", 0, 4)
absences = st.slider("Absences", 0, 50)

# Convert categorical to numeric
gender = 0 if gender == "Male" else 1

# Prepare input
features = np.array([[gender, age, study_time, failures, absences]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("✅ Passed!" if prediction else "❌ Failed")