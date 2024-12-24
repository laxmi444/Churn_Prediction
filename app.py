import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

# load the trained model
model = tf.keras.models.load_model("model.h5")

# loading the encoder and scaler
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)    
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# streamlit app
st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=0, value=600, step=1)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance", min_value=0.0, value=0.0, step=0.01)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])  # 0 = No, 1 = Yes
is_active_member = st.selectbox("Is Active Member", [0, 1])  # 0 = No, 1 = Yes
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=0.0, step=0.01)

#prepare the input data 
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

#encode the 'Geography' column using the loaded one hot encoder
geo_encoded = onehot_encoder_geo.transform(input_data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# encode the 'Gender' column using the label encoder
gender_encoded = label_encoder_gender.transform(input_data[["Gender"]])

# Combine encoded columns with input data
input_data_encoded = input_data.drop(columns=["Geography", "Gender"])  # Remove original columns
input_data_encoded = pd.concat([input_data_encoded, geo_encoded_df], axis=1)
input_data_encoded["Gender"] = gender_encoded

# ensure the columns in 'input_data_encoded' match the order of the columns used to train the scaler
# get the feature names that were used during the fitting of the scaler
scaler_columns = scaler.feature_names_in_

# align the input data columns with the scaler's expected feature names
input_data_encoded = input_data_encoded[scaler_columns]

# scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data_encoded)

#make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn probability: {prediction_proba: .2f}")
#displaying result
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
