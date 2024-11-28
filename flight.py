
import streamlit as st
import pandas as pd
import joblib

st.title("Flight Price Prediction App ✈️")
st.markdown("Upload your model, scaler, and dataset, and provide flight details to predict prices.")

# File upload section
st.sidebar.header("Upload Required Files")
model_file = st.sidebar.file_uploader("Upload the Trained Model (best_regressor.pkl)", type=["pkl"])
scaler_file = st.sidebar.file_uploader("Upload the Scaler (scaler.pkl)", type=["pkl"])
dataset_file = st.sidebar.file_uploader("Upload the Dataset (Clean_Dataset.csv)", type=["csv"])

# Load uploaded files
if model_file and scaler_file and dataset_file:
    try:
        # Load model
        model = joblib.load(model_file)
        st.sidebar.success("Model loaded successfully!")

        # Load scaler
        scaler = joblib.load(scaler_file)
        st.sidebar.success("Scaler loaded successfully!")

        # Load dataset
        df_sample = pd.read_csv(dataset_file)
        st.sidebar.success("Dataset loaded successfully!")

        # Display dataset preview
        st.subheader("Dataset Preview")
        st.write(df_sample.head())

        # Dropdown options from the dataset
        airlines = df_sample['airline'].unique()
        source_cities = df_sample['source_city'].unique()
        destination_cities = df_sample['destination_city'].unique()
        flight_classes = df_sample['class'].unique()
        departure_times = df_sample['departure_time'].unique()

        # User input fields
        st.subheader("Enter Flight Details")
        selected_airline = st.selectbox("Airline", airlines)
        selected_source = st.selectbox("Source City", source_cities)
        selected_destination = st.selectbox("Destination City", destination_cities)
        selected_class = st.selectbox("Class", flight_classes)
        selected_departure_time = st.selectbox("Departure Time", departure_times)
        stops = st.slider("Number of Stops", min_value=0, max_value=3, value=1, step=1)
        duration = st.slider("Flight Duration (in hours)", min_value=1, max_value=24, value=2)
        days_left = st.slider("Days Left for Departure", min_value=1, max_value=365, value=30)

        # Prepare input data
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

        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical features
        categorical_features = ['airline', 'source_city', 'destination_city', 'class', 'departure_time']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_features)

        # Align columns with training data
        missing_cols = set(df_sample.columns) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0
        input_df_encoded = input_df_encoded[df_sample.columns]

        # Scale the input
        input_scaled = scaler.transform(input_df_encoded)

        # Predict price
        if st.button("Predict Price"):
            predicted_price = model.predict(input_scaled)[0]
            st.success(f"The predicted price for your flight is ₹{predicted_price:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload all the required files to proceed.")
