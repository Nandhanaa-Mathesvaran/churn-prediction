import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Bank Customer Churn Prediction")

st.header("Enter Customer Details")

credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=619)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=42)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
has_cr_card = st.number_input("Has Credit Card (0 or 1)", min_value=0, max_value=1, value=1)
is_active_member = st.number_input("Is Active Member (0 or 1)", min_value=0, max_value=1, value=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=101348.90, step=1000.0, format="%.2f")

if st.button("Predict"):
    data = {
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    }

    df_input = pd.DataFrame(data)

    # Encode categorical variables
    mapping_geo = {'France': 0, 'Germany': 1, 'Spain': 2}
    mapping_gender = {'Male': 1, 'Female': 0}

    df_input['Geography'] = df_input['Geography'].map(mapping_geo)
    df_input['Gender'] = df_input['Gender'].map(mapping_gender)

    # Predict
    exit_prob = model.predict_proba(df_input)[0][1]
    stay_prob = 1 - exit_prob
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.success(f"The customer is likely to exit the bank with probability {exit_prob:.2f}")
    else:
        st.success(f"The customer is likely to stay with probability {stay_prob:.2f}")
