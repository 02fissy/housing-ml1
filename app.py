"""Streamlit app to load saved pipeline and predict California house price."""
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title='House Price Predictor', layout='centered')
st.title("Machine Learning Group 1 (Group A)")
st.title('House Price Predictor (California)')
st.write('Input the features and click Predict. The model predicts the median house value (as used in the California Housing dataset).')

# Feature inputs - default values chosen sensibly but user can change
MedInc = st.number_input('Median Income in block (MedInc)', value=3.0, min_value=0.0, step=0.1)
HouseAge = st.number_input('Median House Age (HouseAge)', value=20.0, min_value=0.0, step=1.0)
AveRooms = st.number_input('Average Rooms per household (AveRooms)', value=5.0, min_value=0.0, step=0.1)
AveBedrms = st.number_input('Average Bedrooms per household (AveBedrms)', value=1.0, min_value=0.0, step=0.1)
Population = st.number_input('Population of the block (Population)', value=1000.0, min_value=0.0, step=1.0)
AveOccup = st.number_input('Average occupants per household (AveOccup)', value=3.0, min_value=0.0, step=0.1)
Latitude = st.number_input('Latitude', value=34.0, min_value=32.0, max_value=42.0, step=0.01)
Longitude = st.number_input('Longitude', value=-118.0, min_value=-125.0, max_value=-114.0, step=0.01)

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

if st.button('Predict'):
    try:
        model = joblib.load('best_model.pkl')
    except Exception as e:
        st.error('Could not load model. Make sure you ran `python src/train.py` first to create models/best_model.pkl')
        st.stop()

    pred = model.predict(input_df)[0]
    price = pred * 100000
    st.success(f'Predicted house price: {price:,.3f} (units same as sklearn target)')
   

