# Churn Prediction Model with Streamlit

## Overview
This application is designed to predict customer churn using an Artificial Neural Network (ANN). It utilizes Streamlit for the user interface, providing a simple and interactive experience for users to input data and predict whether a customer will churn.


## Features
- **Input Data**: Users can enter customer data via the app interface.
- **Prediction Model**: An Artificial Neural Network (ANN) model using a **Sequential** architecture.
- **Binary Classification**: The model classifies customers as likely to churn (1) or not likely to churn (0).
- **Optimizer & Loss Function**: The model uses the Adam optimizer and Binary Cross-Entropy loss function.
- **One-Hot Encoding**: Categorical features are preprocessed using One-Hot Encoding for efficient model performance.

## Technologies
- **Streamlit**: For creating the interactive web interface.
- **TensorFlow/Keras**: For building the Artificial Neural Network (ANN) using the Sequential model.
- **Scikit-Learn**: For data preprocessing (One-Hot Encoding).
- **Pandas**: For handling and manipulating data.
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


### Running the Streamlit App

To run the Streamlit app, execute the following command in your terminal:
```bash
streamlit run app.py

