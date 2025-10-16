import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("Bank Customer Churn Prediction")

with open("decision_tree_model.pkl", "rb") as file:
    model = pickle.load(file)

st.header("Enter Customer Details")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.checkbox("Has Credit Card")  
is_active_member = st.checkbox("Is Active Member")  
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict Churn"):
    mapping_geo = {"France": 0, "Germany": 1, "Spain": 2}
    mapping_gender = {"Male": 1, "Female": 0}

    input_dict = {
        "CreditScore": credit_score,
        "Geography": mapping_geo[geography],
        "Gender": mapping_gender[gender],
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"The customer is likely to exit with probability {probability:.2f}")
    else:
        st.info(f"The customer is likely to stay with probability {1-probability:.2f}")
