"""Streamlit app to load saved pipeline and predict California house price."""
import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title='House Price Predictor', layout='centered')
st.title("Machine Learning Group 1 (Group A)")
st.header('House Price Predictor (California)')
st.write('Input the features and click Predict. The model predicts the median house value in units of $100,000.')

# Feature inputs
MedInc = st.number_input('Median Income in block (MedInc)', value=3.0, min_value=0.0, step=0.1)
HouseAge = st.number_input('Median House Age (HouseAge)', value=20.0, min_value=0.0, step=1.0)
AveRooms = st.number_input('Average Rooms per household (AveRooms)', value=5.0, min_value=0.0, step=0.1)
AveBedrms = st.number_input('Average Bedrooms per household (AveBedrms)', value=1.0, min_value=0.0, step=0.1)
Population = st.number_input('Population of the block (Population)', value=1000.0, min_value=0.0, step=1.0)
AveOccup = st.number_input('Average occupants per household (AveOccup)', value=3.0, min_value=0.0, step=0.1)
Latitude = st.number_input('Latitude', value=34.0, min_value=32.0, max_value=42.0, step=0.01)
Longitude = st.number_input('Longitude', value=-118.0, min_value=-125.0, max_value=-114.0, step=0.01)

# Build input dataframe
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

st.write("Input Summary")
st.dataframe(input_df)

if st.button('Predict'):
    model_path = 'models/best_model.pkl'

   
    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`.\n\n"
                 "Run `python src/train.py` to generate it.")
        st.stop()

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f'Error loading model: {e}')
        st.stop()

    # Prediction
    pred = model.predict(input_df)[0]
    price = pred * 100000

    st.success(f'Predicted house price: **${price:,.2f}**')
    st.caption("(*The model predicts median house price × 100,000 as used in the California Housing dataset.*)")
