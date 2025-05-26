import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

st.write("Enter the following details:")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 10, 100, step=1)

# Predict
if st.button("Predict"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is not likely to have diabetes.")
