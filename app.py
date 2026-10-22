import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction System")
st.write("Enter the four attributes below to get an educational model prediction.")

MODEL_PATH = "heart_disease_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the Jupyter Notebook first to create heart_disease_model.pkl, then place it beside app.py.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
sex_label = st.selectbox("Sex", ["Female (0)", "Male (1)"])
cp = st.selectbox(
    "Chest pain type (cp)",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Typical angina",
        2: "2 - Atypical angina",
        3: "3 - Non-anginal pain",
        4: "4 - Asymptomatic"
    }[x]
)
thalach = st.number_input("Maximum heart rate achieved (thalach)", min_value=40, max_value=250, value=160)

if st.button("Predict"):
    inputs = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex_label.startswith("Male") else 0,
        "cp": cp,
        "thalach": thalach
    }])
    prediction = int(model.predict(inputs)[0])
    probability = float(model.predict_proba(inputs)[0][1])

    if prediction == 1:
        st.warning("Model prediction: Heart disease indicators detected.")
    else:
        st.success("Model prediction: No heart disease indicators detected.")

    st.write(f"Model estimated probability for the positive class: {probability:.1%}")

st.caption("Educational demonstration only. This model is not a medical diagnostic tool. Consult a qualified healthcare professional for medical concerns.")
