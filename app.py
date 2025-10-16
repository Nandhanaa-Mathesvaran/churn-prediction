import streamlit as st
import pandas as pd
import pickle

with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction App")

st.write("Enter customer details below to predict whether they are likely to exit the bank.")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

if st.button("Predict"):
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    df_input = pd.DataFrame([data])
    mapping_geo = {'France': 0, 'Germany': 1, 'Spain': 2}
    mapping_gender = {'Female': 0, 'Male': 1}
    df_input['Geography'] = df_input['Geography'].map(mapping_geo)
    df_input['Gender'] = df_input['Gender'].map(mapping_gender)

    required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                     'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    df_input = df_input[required_cols]

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.write(f"Customer is likely to LEAVE the bank. Exit Probability: {probability:.2f}")
    else:
        st.write(f"Customer is likely to STAY with the bank. Exit Probability: {probability:.2f}")
