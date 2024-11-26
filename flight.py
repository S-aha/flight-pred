import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the app
st.set_page_config(page_title="Flight Price Prediction", layout="wide")

# Initialize session state for the uploaded file
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Sidebar navigation
st.sidebar.header("Navigation")
tabs = st.sidebar.radio("Choose a section", ["Home", "Prediction", "Comparison"])

# Home tab: Upload dataset
if tabs == "Home":
    st.title("Flight Price Prediction App")
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        # Save the uploaded file to session state
        st.session_state.uploaded_file = uploaded_file

        # Read the uploaded CSV file into a DataFrame
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(data.head())
            st.write("Dataset Shape:", data.shape)

            # Preprocessing sample
            st.subheader("Preprocessing Insights")
            if "Departure_Date" not in data.columns:
                st.warning("Adding a simulated 'Departure_Date' column.")
                data["Departure_Date"] = pd.to_datetime(
                    np.random.choice(pd.date_range("2023-12-01", "2024-01-31"), len(data))
                )
            st.write("Sample Data with 'Departure_Date':")
            st.dataframe(data[["Departure_Date"]].head())
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

# Prediction tab
elif tabs == "Prediction":
    st.header("Predict Flight Prices")
    st.write("Enter your travel details below for prediction:")
    
    # Input fields for user data
    booking_date = st.date_input("Booking Date")
    departure_date = st.date_input("Departure Date")
    num_passengers = st.number_input("Number of Passengers", min_value=1, step=1)
    
    # Traveler categorization
    if booking_date and departure_date:
        days_diff = (departure_date - booking_date).days
        if days_diff > 60:
            category = "Early Bird"
        elif 30 < days_diff <= 60:
            category = "Planner"
        elif 7 < days_diff <= 30:
            category = "Last Minute Planner"
        else:
            category = "Spontaneous Traveler"
        st.write(f"Traveler Type: **{category}**")

    # Prediction button
    if st.button("Predict Price"):
        # Dummy prediction logic (replace with actual model predictions)
        predicted_price = np.random.randint(3000, 15000)
        st.success(f"Predicted Flight Price: ₹{predicted_price}")

# Comparison tab
elif tabs == "Comparison":
    st.header("Compare Flight Prices Across Airlines")
    
    if st.session_state.uploaded_file:
        # Use the uploaded dataset from session state
        try:
            data = pd.read_csv(st.session_state.uploaded_file)
            
            if "airline" in data.columns and "price" in data.columns:
                st.write("price Comparison by airline")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x="airline", y="price", data=data, ax=ax)
                ax.set_title("Flight Price Distribution Across Airlines")
                st.pyplot(fig)
            else:
                st.error("Dataset must contain 'airline' and 'price' columns for comparison.")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("Please upload a dataset in the Home section to enable comparison.")




    
