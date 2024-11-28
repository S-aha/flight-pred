import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and description
st.title("Flight Price Prediction App ✈️")
st.markdown("Predict the price of your flight by selecting the relevant options below.")

# Load preprocessed data and models
@st.cache
def load_model_and_scaler():
    model = joblib.load('best_regressor.pkl')  # Trained model
    scaler = joblib.load('scaler.pkl')         # Scaler
    df_sample = pd.read_csv('Clean_Dataset.csv')  # Dataset for dropdown values
    return model, scaler, df_sample

model, scaler, df_sample = load_model_and_scaler()

# Dropdown options based on the dataset
airlines = df_sample['airline'].unique()
source_cities = df_sample['source_city'].unique()
destination_cities = df_sample['destination_city'].unique()
flight_classes = df_sample['class'].unique()
departure_times = df_sample['departure_time'].unique()

# Input fields
st.sidebar.header("Select Flight Details:")
selected_airline = st.sidebar.selectbox("Airline", airlines)
selected_source = st.sidebar.selectbox("Source City", source_cities)
selected_destination = st.sidebar.selectbox("Destination City", destination_cities)
selected_class = st.sidebar.selectbox("Class", flight_classes)
selected_departure_time = st.sidebar.selectbox("Departure Time", departure_times)

# Numeric inputs
stops = st.sidebar.slider("Number of Stops", min_value=0, max_value=3, value=1, step=1)
duration = st.sidebar.slider("Flight Duration (in hours)", min_value=1, max_value=24, value=2)
days_left = st.sidebar.slider("Days Left for Departure", min_value=1, max_value=365, value=30)

# Prepare the input for prediction
input_data = {
    'airline': selected_airline,
    'source_city': selected_source,
    'destination_city': selected_destination,
    'class': selected_class,
    'departure_time': selected_departure_time,
    'stops': stops,
    'duration': duration,
    'days_left': days_left
}

# Convert to DataFrame for processing
input_df = pd.DataFrame([input_data])

# One-hot encode the categorical features (align with the training process)
categorical_features = ['airline', 'source_city', 'destination_city', 'class', 'departure_time']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_features)

# Align the columns with the training dataset
missing_cols = set(df_sample.columns) - set(input_df_encoded.columns)
for col in missing_cols:
    input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[df_sample.columns]

# Scale the input
input_scaled = scaler.transform(input_df_encoded)

# Predict the price
if st.sidebar.button("Predict Price"):
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"The predicted flight price is ₹{predicted_price:.2f}")
    import os
import joblib
import pandas as pd

@st.cache
def load_model_and_scaler():
    base_path = os.path.dirname(__file__)  # Get the directory of the script
    model = joblib.load(os.path.join(base_path, 'best_regressor.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    df_sample = pd.read_csv(os.path.join(base_path, 'Clean_Dataset.csv'))
    return model, scaler, df_sample
    import os
print(os.getcwd())
try:
    model = joblib.load('best_regressor.pkl')
    scaler = joblib.load('scaler.pkl')
    df_sample = pd.read_csv('Clean_Dataset.csv')
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")


