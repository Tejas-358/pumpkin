import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')


# Function to make predictions
def make_prediction(input_data):
    # Convert input_data into a DataFrame (assuming input is a dictionary)
    df = pd.DataFrame([input_data])

    # Make a prediction using the pre-trained model
    prediction = model.predict(df)

    return prediction[0]


# Streamlit UI
st.title("Random Forest Classifier for Pumpkin Seed")

st.write("""
Enter Values
""")

# Input fields for each feature in the dataset
area = st.number_input("Area", min_value=0.0, value=56276.0)  # Ensure the value is a float
perimeter = st.number_input("Perimeter", min_value=0.0, value=888.242)
major_axis_length = st.number_input("Major Axis Length", min_value=0.0, value=326.1485)
minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0, value=220.2388)
convex_area = st.number_input("Convex Area", min_value=0.0, value=56831.0)  # Ensure the value is a float
equiv_diameter = st.number_input("Equiv Diameter", min_value=0.0, value=267.6805)
eccentricity = st.number_input("Eccentricity", min_value=0.0, value=0.7376)
solidity = st.number_input("Solidity", min_value=0.0, value=0.9902)
extent = st.number_input("Extent", min_value=0.0, value=0.7453)
roundness = st.number_input("Roundness", min_value=0.0, value=0.8963)
aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, value=1.4809)
compactness = st.number_input("Compactness", min_value=0.0, value=0.8207)

# Dictionary to hold the input data
input_data = {
    'Area': area,
    'Perimeter': perimeter,
    'Major_Axis_Length': major_axis_length,
    'Minor_Axis_Length': minor_axis_length,
    'Convex_Area': convex_area,
    'Equiv_Diameter': equiv_diameter,
    'Eccentricity': eccentricity,
    'Solidity': solidity,
    'Extent': extent,
    'Roundness': roundness,
    'Aspect_Ration': aspect_ratio,
    'Compactness': compactness
}

# Button to make the prediction
if st.button('Predict'):
    # Get the prediction
    prediction = make_prediction(input_data)

    # Display the result
    st.write(f"The predicted class is: {prediction}")
