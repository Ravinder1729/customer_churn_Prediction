import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import OneHotEncoder
st.image(r"C:\Users\ravin\Downloads\innomatics-footer-logo (1).webp")
st.title("Customer Churn Prediction")

# Load model
log = pickle.load(open(r"D:\churn\bestmodel2.pkl", 'rb'))


# User input via radio buttons
InternetService = st.radio("Choose whether customer has Internet service or not:",
                           options=['DSL', 'Fiber optic', 'No'])

OnlineSecurity = st.radio("Choose whether customer has OnlineSecurity service or not:",
                           options=['Yes', 'No'])

OnlineBackup = st.radio("Choose whether customer has OnlineBackup or not:",
                           options=['Yes', 'No'])

TechSupport = st.radio("Choose whether customer has TechSupport or not:",
                           options=['Yes', 'No'])

Contract = st.radio("Choose customer has which Contract type:",
                    options=['Month-to-month', 'One year', 'Two year'])

PaymentMethod = st.radio("Choose customer has which PaymentMethod type:",
                         options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                  'Credit card (automatic)'])

# Create a DataFrame from user input
user_input = {
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'TechSupport': [TechSupport],
    'Contract': [Contract],
    'PaymentMethod': [PaymentMethod]
}
user_input_df = pd.DataFrame(user_input,columns=['InternetService','OnlineSecurity','OnlineBackup','TechSupport','Contract','PaymentMethod'])

oe = pickle.load(open(r"D:\churn\oe.pkl", 'rb'))
d = oe.transform(user_input_df)
# Predict using the model
def predict_churn(d):
    result = log.predict(d)
    return result[0]
if st.button("Predict"):
    prediction = predict_churn(d)
    if prediction == 'Yes':
        st.write("Customer is Churn")
    else:
        st.write("Customer is Unchurn")