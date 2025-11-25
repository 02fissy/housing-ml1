import streamlit as st
import joblib
import pandas as pd
import os
import urllib.request

MODEL_PATH = "models/best_model.pkl"
MODEL_URL = "https://raw.githubusercontent.com/02fissy/housing-ml1/main/models/best_model.pkl"

# Create folder if not exists
os.makedirs("models", exist_ok=True)

# Download model if not already downloaded
if not os.path.exists(MODEL_PATH):
    try:
        st.write("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")

# Try loading model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model file: {e}")
    st.stop()

st.set_page_config(page_title='California House Price Predictor', layout='centered')
st.title('California House Price Predictor')

# Feature inputs
MedInc = st.number_input('Median Income (MedInc)', value=3.0, min_value=0.0, step=0.1)
HouseAge = st.number_input('Median House Age (HouseAge)', value=20.0, min_value=0.0, step=1.0)
AveRooms = st.number_input('Average Rooms (AveRooms)', value=5.0, min_value=0.0, step=0.1)
AveBedrms = st.number_input('Average Bedrooms (AveBedrms)', value=1.0, min_value=0.0, step=0.1)
Population = st.number_input('Population', value=1000.0, min_value=0.0, step=1.0)
AveOccup = st.number_input('Average Occupancy (AveOccup)', value=3.0, min_value=0.0, step=0.1)
Latitude = st.number_input('Latitude', value=34.0, min_value=32.0, max_value=42.0, step=0.01)
Longitude = st.number_input('Longitude', value=-118.0, min_value=-125.0, max_value=-114.0, step=0.01)

# Build DataFrame
input_df = pd.DataFrame([{
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude
}])

# Predict button
if st.button('Predict'):
    pred = model.predict(input_df)[0]
    price_dollars = pred * 100000
    st.success(f'Predicted House Price: ${price_dollars:,.2f}')
