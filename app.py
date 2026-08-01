import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('knn_regressor.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Insurance Charge Predictor")
st.write("Enter patient details to estimate medical insurance charges.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

if st.button("Predict Charges"):
    # Prepare the input for One-Hot Encoding (matching the training format)
    data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    
    input_df = pd.DataFrame([data])
    
    # Scale the input
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)
    
    st.success(f"Estimated Insurance Charges: ${prediction[0]:,.2f}")