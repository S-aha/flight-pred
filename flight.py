# Comparison tab
elif tabs == "Comparison":
    st.header("Compare Flight Prices Across Airlines")
    
    if st.session_state.uploaded_file:
        try:
            # Reset file pointer and read the file named "flights"
            st.session_state.uploaded_file.seek(0)  # Ensure the file pointer is at the beginning
            data = pd.read_csv(st.session_state.uploaded_file)

            # Debugging: Display dataset details
            st.write("Dataset Details:")
            st.write(f"File Name: flights")
            st.write(f"Number of Rows: {data.shape[0]}, Number of Columns: {data.shape[1]}")

            # Check for required columns
            if data.empty:
                st.error("The uploaded file 'flights' is empty. Please upload a valid file.")
            elif "Airline" in data.columns and "Price" in data.columns:
                # Generate the boxplot for comparison
                st.write("Price Comparison by Airline")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x="Airline", y="Price", data=data, ax=ax)
                ax.set_title("Flight Price Distribution Across Airlines")
                st.pyplot(fig)
            else:
                # Report missing columns
                missing_cols = [col for col in ["Airline", "Price"] if col not in data.columns]
                st.error(f"The dataset 'flights' is missing required columns: {', '.join(missing_cols)}.")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file 'flights' is empty or invalid. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred while processing the file 'flights': {e}")
    else:
        st.warning("Please upload the dataset named 'flights' in the Home section to enable comparison.")




    
