import time
import streamlit as st
import pandas as pd 
from geopy.geocoders import Nominatim
from src.prediction_pipeline import PredictionPipeline

def address_to_latlon(address):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None




def prediction_page():
    st.title("Prediction")

    # Ensure model and preprocessor are loaded
    if 'pipeline' not in st.session_state:
        st.warning("Please upload and train the model first.")
        return
    
    pipeline = st.session_state['pipeline']

    # User input for prediction
    st.write("Enter beneficiary details for prediction")
    
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0)
    height = st.number_input("Height (cm)", min_value=0)
    ethnicity = st.selectbox("Ethnicity", 
        ['Italian', 'Japanese', 'Mexican', 'Korean', 'Spanish', 'Polish',
         'Canadian', 'Greek', 'Brazilian', 'Chinese', 'Irish', 'English',
         'American', 'Dutch', 'Indian', 'French', 'Vietnamese', 'Russian',
         'German', 'Swedish'])
    diabetic = st.selectbox("Diabetic", ["Yes", "No"])
    religion = st.selectbox("Religion", 
        ['Deism', 'Islam', 'Druidism', 'Paganism', 'Zoroastrianism',
         'Atheism', 'Hinduism', 'Baháʼí', 'Sikhism', 'Buddhism',
         'Unitarianism', 'Confucianism', 'Jainism', 'Pastafarianism',
         'Shinto', 'Christianity', 'Agnostic', 'Judaism', 'Rastafarianism',
         'Taoism'])
    
    address_box = st.selectbox("Select address or Latitude and Longitude",
                               ["Address", "Latitude & Longitude"])
    
    if address_box == "Address":
        address = st.text_input("Address")
        # latitude, longitude = address_to_latlon(address)
        

    else:
        latitude = st.number_input("Latitude")
        longitude = st.number_input("Longitude")
        
    
    if st.button("Predict"):
        if latitude is None or longitude is None:
            st.error("Invalid address")
            return
        
        prediction_pipeline = PredictionPipeline()
        
        
        # Create dataframe for prediction
        input_data = {
            'Name': name,
            'Age': age,
            'Height': height,
            'Ethnicity': ethnicity,
            'Diabetic': diabetic,
            'Religion': religion,
            'latitude': latitude,
            'longitude': longitude
        }
        
        st.write("Your input Data preview:", pd.DataFrame([input_data]).head())
        time.sleep(1)
        # Preprocess input data
        st.write("Fitting input data for prediction")
        
        # Prediction
        st.write("Predicting beneficiary cluster")
        
        match_data = prediction_pipeline.match(
            input_df= input_data
        )
        
        st.success(f"The probable benificiaries are predicted. ")
        
        st.dataframe(match_data)
        
        
prediction_page()