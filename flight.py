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
        st.session_state.uploaded_file = uploaded_file  # Save to session state
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
        # Use the uploaded dataset from session 
        data = pd.read_csv(st.session_state.uploaded_file)
        if "Airline" in data.columns and "Price" in data.columns:
            st.write("Price Comparison by Airline")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Airline", y="Price", data=data, ax=ax)
            ax.set_title("Flight Price Distribution Across Airlines")
            st.pyplot(fig)
        else:
            st.error("Dataset must contain 'Airline' and 'Price' columns for comparison.")
    else:
        st.warning("Please upload a dataset in the Home section to enable comparison.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
@st.cache
def load_data():
    return pd.read_csv("Clean_Dataset.csv")

# Categorize flights
def categorize_flights(data):
    data['Departure_Hour'] = pd.to_datetime(data['Departure_Time']).dt.hour
    data['Shift'] = data['Departure_Hour'].apply(lambda x: 'Morning' if 6 <= x < 18 else 'Evening')
    return data

# Price categorization for calendar
def categorize_price(data):
    price_bins = pd.qcut(data['Price'], q=3, labels=['Low', 'Moderate', 'High'])
    data['Price_Category'] = price_bins
    return data

# Streamlit App
def main():
    st.title("Flight Price Prediction")
    st.sidebar.title("Flight Price Analysis")

    # Load and preprocess data
    data = load_data()
    data = categorize_flights(data)
    data = categorize_price(data)

    # Display dataset
    if st.sidebar.checkbox("Show Dataset"):
        st.write(data)

    # Early bird vs. last-minute
    st.header("Early Bird vs. Last-Minute Pricing")
    days_to_departure = (pd.to_datetime(data['Departure_Date']) - datetime.now()).dt.days
    data['Booking_Category'] = days_to_departure.apply(lambda x: 'Early Bird' if x > 30 else 'Last Minute')
    booking_stats = data.groupby('Booking_Category')['Price'].mean()
    st.bar_chart(booking_stats)

    # Flight price comparison
    st.header("Flight Price Comparison by Airline")
    airline_stats = data.groupby('Airline')['Price'].mean().sort_values()
    st.bar_chart(airline_stats)

    # Morning vs. Evening Shifts
    st.header("Shift Analysis (Morning vs. Evening)")
    shift_stats = data.groupby('Shift')['Price'].mean()
    st.bar_chart(shift_stats)

    # Calendar visualization
    st.header("Price Calendar")
    calendar_data = data[['Departure_Date', 'Price_Category']].drop_duplicates()
    calendar_chart_data = calendar_data.pivot_table(index='Departure_Date', columns='Price_Category', aggfunc='size', fill_value=0)
    st.line_chart(calendar_chart_data)

if __name__ == "__main__":
    main()



       
    
