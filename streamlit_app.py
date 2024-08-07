import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the trained model, transformer, and scaler
dummy_transformer = joblib.load('dummy_transformer.pkl')
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define a function to process input data
def process_input(data):
    # Convert categorical features to dummy variables
    data_dummies = dummy_transformer.transform(data)
    # Ensure all columns used in training are present
    data_dummies = data_dummies.reindex(columns=scaler.feature_names_in_, fill_value=0)
    # Scale features
    data_scaled = scaler.transform(data_dummies)
    return data_scaled

# Streamlit app
st.title('Credit Risk Prediction App')

st.write('Enter the following details to predict credit risk:')

# Collect user input
# Adjust these inputs based on your dataset
age = st.number_input('Age', min_value=18, max_value=100, value=30)
annual_income = st.number_input('Annual Income', min_value=1000, max_value=1000000, value=50000)
monthly_inhand_salary = st.number_input('Monthly In-hand Salary', min_value=100, max_value=100000, value=4000)
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, max_value=20, value=2)
num_credit_card = st.number_input('Number of Credit Cards', min_value=0, max_value=20, value=1)
interest_rate = st.number_input('Interest Rate', min_value=0.0, max_value=50.0, value=15.0)
num_of_loan = st.number_input('Number of Loans', min_value=0, max_value=20, value=1)
type_of_loan = st.selectbox('Type of Loan', ['Auto Loan', 'Home Loan', 'Personal Loan', 'Credit Card Loan'])
delay_from_due_date = st.number_input('Delay from Due Date', min_value=0, max_value=100, value=5)
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, max_value=100, value=2)
changed_credit_limit = st.number_input('Changed Credit Limit', min_value=0, max_value=100, value=5)
num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=20, value=2)
credit_mix = st.selectbox('Credit Mix', ['Good', 'Bad', 'Standard'])
outstanding_debt = st.number_input('Outstanding Debt', min_value=0.0, max_value=1000000.0, value=5000.0)
credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, max_value=100.0, value=30.0)
credit_history_age = st.number_input('Credit History Age', min_value=0, max_value=100, value=10)
payment_of_min_amount = st.selectbox('Payment of Minimum Amount', ['Yes', 'No'])
total_emi_per_month = st.number_input('Total EMI per Month', min_value=0.0, max_value=10000.0, value=500.0)
amount_invested_monthly = st.number_input('Amount Invested Monthly', min_value=0.0, max_value=10000.0, value=100.0)
payment_behaviour = st.selectbox('Payment Behaviour', ['Prompt', 'Late', 'Partially Paid'])
monthly_balance = st.number_input('Monthly Balance', min_value=-10000.0, max_value=100000.0, value=1000.0)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Age': [age],
    'Annual_Income': [annual_income],
    'Monthly_Inhand_Salary': [monthly_inhand_salary],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_card],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [num_of_loan],
    'Type_of_Loan': [type_of_loan],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed_payment],
    'Changed_Credit_Limit': [changed_credit_limit],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Credit_Mix': [credit_mix],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Credit_History_Age': [credit_history_age],
    'Payment_of_Min_Amount': [payment_of_min_amount],
    'Total_EMI_per_month': [total_emi_per_month],
    'Amount_invested_monthly': [amount_invested_monthly],
    'Payment_Behaviour': [payment_behaviour],
    'Monthly_Balance': [monthly_balance]
})

# Process the input data
input_data_processed = process_input(input_data)

# Predict using the loaded model
if st.button('Predict'):
    prediction = xgb_model.predict(input_data_processed)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    st.write(f'The predicted credit score category is: **{predicted_class}**')






