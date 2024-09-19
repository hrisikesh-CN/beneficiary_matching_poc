import time
import streamlit as st
import pandas as pd 
import warnings
from src.prediction_pipeline import PredictionPipeline
from src.components.match_finder import MatchFinderChatbot






st.title("Prediction")
st.session_state["prediction_completed"] = False
match_finder = MatchFinderChatbot()


def prediction():
    # Ensure model and preprocessor are loaded
    if 'pipeline' not in st.session_state:
        st.warning("Please upload data and train the model first.")
        st.stop()

    pipeline = st.session_state['pipeline']

    # User input for prediction
    st.write("Enter beneficiary details for prediction")

    with st.form("input_form"):

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

        # address_box = st.selectbox("Select address or Latitude and Longitude",
                                    # ["Address", "Latitude & Longitude"])

        # if address_box == "Address":
        #     address = st.text_input("Address")
        #     # latitude, longitude = address_to_latlon(address)
            

        # else:
        latitude = st.number_input("Latitude")
        longitude = st.number_input("Longitude")
        

        submitted = st.form_submit_button("Submit")
        if submitted:
            if latitude is None or longitude is None:
                st.error("Invalid address")
                st.stop()
        
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
            input_dataframe = pd.DataFrame([input_data]).head()
            
            st.session_state["input_dataframe"] = input_dataframe
            
            st.write("Your input Data preview:", input_dataframe)
            time.sleep(1)
            # Preprocess input data
            st.write("Fitting input data for prediction")
            
            # Prediction
            st.write("Predicting beneficiary cluster")
            
            match_data = prediction_pipeline.match(
                input_df= input_data
            )
            
            st.session_state["match_data"] = match_data
            
            st.success(f"The probable benificiaries are predicted. ")
            
            st.dataframe(match_data)
            
            st.session_state["prediction_completed"] = True
    
    

if not st.session_state["prediction_completed"]:
    prediction()    
    
if st.session_state["prediction_completed"]:
    with st.sidebar:
        # if st.button("Get the review from llm"):
        input_data = st.session_state["input_dataframe"]
        matched_data = st.session_state["match_data"] 
        explanation = match_finder.find_best_match(input_data, matched_data)
        st.write(explanation)
        st.stop()
        
        


