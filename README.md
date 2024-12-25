# Churn Prediction Model with Streamlit

This project involves building a churn prediction model using machine learning and deploying it as a web application using Streamlit.

## Project Overview

The goal of this project is to predict whether a customer will churn (leave a service) based on various features like customer behavior, account details, and historical data. The model is trained using a classification algorithm and then deployed using Streamlit for easy interaction.

### Key Features:
- **Churn Prediction Model:** Predicts whether a customer will churn or stay based on historical data.
- **Streamlit Web App:** A user-friendly interface where you can input customer data and get churn predictions.

## Installation

Follow these steps to set up the project on your local machine.

1. Clone this repository:
    ```bash
    git clone https://github.com/laxmi444/Churn_Prediction-streamlit.git
    cd Churn_Prediction-streamlit
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

## Model Development

### Data Preprocessing

The dataset is first preprocessed to handle missing values, encode categorical features, and normalize numerical values. Various machine learning models like Logistic Regression, Random Forest, or XGBoost are tested to find the best performing model.

### Model Training

- The model is trained using `scikit-learn`.
- Cross-validation is applied to evaluate the model's performance.
- Hyperparameters are tuned for better accuracy.

## Streamlit Web App

Once the model is trained, we create a simple web app using Streamlit to interact with the model. The app allows users to input customer information and get a prediction on whether they will churn or not.

### Running the Streamlit App

To run the Streamlit app, execute the following command in your terminal:
```bash
streamlit run app.py

