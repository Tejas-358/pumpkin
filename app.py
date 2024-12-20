import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from azure.storage.blob import BlobServiceClient
import joblib
import os
import io

# Azure Blob Storage credentials and settings
connection_string = "DefaultEndpointsProtocol=https;AccountName=storagepumpkin;AccountKey=H0sN/xHT0FSsLv2RofJQ+oHu0nEn+LABeoeaL80zmfW1fsrLBW7Ia77BYFyE2nWuQAKH0FnqwhyG+AStIxDkBw==;EndpointSuffix=core.windows.net"
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")

container_name = "dataset"
blob_name = "logistic_regression_model.pkl"

def load_model_from_blob():
    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the blob as a byte stream
    download_stream = blob_client.download_blob()
    model_data = download_stream.readall()

    # Load the model using joblib from the byte stream
    model = joblib.load(io.BytesIO(model_data))
    return model

# Load model
model = load_model_from_blob()


# Function to make predictions
def make_prediction(input_data):
    # Convert input_data into a DataFrame (assuming input is a dictionary)
    df = pd.DataFrame([input_data])

    # Make a prediction using the pre-trained model
    prediction = model.predict(df)

    return prediction[0]


# Streamlit UI
st.title("Linear Regressor Classifier for Pumpkin Seed")

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
