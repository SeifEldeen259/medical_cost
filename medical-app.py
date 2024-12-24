import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# Page setting
st.set_page_config(page_title="Medical Costs Prediction App", layout="centered", page_icon="📊")

st.title("MEDICAL COSTS PREDICTION")
st.image('health-costs.jpg')
st.text("Fill-in the following values to predict the costs of your medical treatment")

age = st.slider('Age', 18, 65)
bmi = st.number_input('BMI')
child = st.number_input('No. of Children')
sex = st.selectbox('Gender', ['Male', 'Female'])
smoker = st.selectbox('Smoking', ['Smoker', 'Non-Smoker'])
region = st.radio('Region', ['South East', 'North East', 'South West', 'North West'])
btn = st.button("Submit")


# Launching the App
if btn == True:

    # loading model and scalers
    scaler = joblib.load('scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    model = joblib.load('model.pkl')

    # encoding and scaling input
    sex_mapping = {'Male': 1, 'Female': 0}
    smoker_mapping = {'Smoker': 1, 'Non-Smoker': 0}
    region_mapping = {'South East': 2, 'North East': 3, 'South West': 2, 'North West': 1}

    sex_encoded = sex_mapping[sex]
    smoker_encoded = smoker_mapping[smoker]
    region_encoded = region_mapping[region]

    input_data = np.array([[age, bmi, child, sex_encoded, smoker_encoded, region_encoded]])
    input_data_scaled = scaler.transform(input_data)

    prediction_scaled = model.predict(input_data_scaled)
    prediction_original = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    st.success(prediction_scaled)