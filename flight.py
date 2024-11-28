import streamlit as st
import pandas as pd
import joblib

# Title and description
st.title("Flight Price Prediction App ✈️")
st.markdown("Upload your model, scaler, and dataset files below to predict flight prices.")

# File upload widgets
uploaded_model = st.file_uploader("Upload Trained Model (`best_regressor.pkl`):", type=["pkl"])
uploaded_scaler = st.file_uploader("Upload Scaler (`scaler.pkl`):
