import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from textblob import TextBlob  # For sentiment analysis

# Load data
@st.cache
def load_data():
    data = pd.read_csv("Clean_Dataset.csv")
    # Adding simulated ratings and reviews
    data['Ratings'] = np.random.randint(1, 6, size=len(data))  # Ratings: 1 to 5
    data['Reviews'] = np.random.choice(
        ["Great experience", "Average service", "Not worth the price",
         "Excellent value for money", "Poor customer support"], len(data)
    )
    # Simulate additional costs
    data['Baggage_Fee'] = np.random.randint(0, 51, size=len(data))  # $0 - $50
    data['Seat_Selection_Fee'] = np.random.randint(0, 21, size=len(data))  # $0 - $20
    return data

# Categorize flights
def categorize_flights(data):
    data['Departure_Hour'] = pd.to_datetime(data['Departure_Time']).dt.hour
    data['Shift'] = data['Departure_Hour'].apply(lambda x: 'Morning' if 6 <= x < 18 else 'Evening')
    return data

# Dynamic filters
def filter_data(data, airline, shift, departure_city, destination_city, price_range):
    filtered_data = data[
        (data['Airline'] == airline if airline else True) &
        (data['Shift'] == shift if shift else True) &
        (data['Departure_City'] == departure_city if departure_city else True) &
        (data['Destination_City'] == destination_city if destination_city else True) &
        (data['Price'] >= price_range[0]) & (data['Price'] <= price_range[1])
    ]
    return filtered_data

# Sentiment analysis
def sentiment_analysis(reviews):
    sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
    return ["Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral" for sentiment in sentiments]

# Streamlit App
def main():
    st.title("Advanced Flight Price Prediction and Analysis")
    st.sidebar.title("Filters and Options")

    # Load data
    data = load_data()
    data = categorize_flights(data)

    # Sidebar filters
    airlines = st.sidebar.selectbox("Select Airline", options=[None] + list(data['Airline'].unique()))
    shift = st.sidebar.selectbox("Select Flight Shift", options=[None, "Morning", "Evening"])
    departure_city = st.sidebar.selectbox("Select Departure City", options=[None] + list(data['Departure_City'].unique()))
    destination_city = st.sidebar.selectbox("Select Destination City", options=[None] + list(data['Destination_City'].unique()))
    price_range = st.sidebar.slider("Select Price Range", int(data['Price'].min()), int(data['Price'].max()), (int(data['Price'].min()), int(data['Price'].max())))

    # Filter data based on selections
    filtered_data = filter_data(data, airlines, shift, departure_city, destination_city, price_range)
    st.write(f"### Filtered Results ({len(filtered_data)} flights)")
    st.write(filtered_data)

    # Ratings and reviews
    st.header("Airline Ratings and Reviews")
    avg_ratings = filtered_data.groupby('Airline')['Ratings'].mean().sort_values()
    st.bar_chart(avg_ratings)
    filtered_data['Sentiment'] = sentiment_analysis(filtered_data['Reviews'])
    st.write("Sentiments of Reviews:")
    st.write(filtered_data[['Airline', 'Reviews', 'Sentiment']])

    # Additional costs visualization
    st.header("Additional Costs Breakdown")
    additional_costs = filtered_data[['Airline', 'Baggage_Fee', 'Seat_Selection_Fee']].groupby('Airline').mean()
    st.bar_chart(additional_costs)

    # Personalized recommendations
    st.header("Personalized Recommendations")
    cheapest_flight = filtered_data.loc[filtered_data['Price'].idxmin()] if not filtered_data.empty else None
    if cheapest_flight is not None:
        st.subheader("Cheapest Flight")
        st.write(f"**Airline**: {cheapest_flight['Airline']}")
        st.write(f"**Price**: ${cheapest_flight['Price']}")
        st.write(f"**Shift**: {cheapest_flight['Shift']}")
        st.write(f"**Departure City**: {cheapest_flight['Departure_City']}")
        st.write(f"**Destination City**: {cheapest_flight['Destination_City']}")
    else:
        st.write("No flights available with the current filters.")

    # Seasonal insights
    st.header("Seasonal Insights")
    seasonal_data = data.copy()
    seasonal_data['Month'] = pd.to_datetime(seasonal_data['Departure_Date']).dt.month
    avg_monthly_price = seasonal_data.groupby('Month')['Price'].mean()
    st.line_chart(avg_monthly_price)

if __name__ == "__main__":
    main()
