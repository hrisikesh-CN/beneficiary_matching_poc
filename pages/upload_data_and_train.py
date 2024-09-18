from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
import streamlit as st 
import pandas as pd 

st.title("Upload Data and Train")

# File uploader
file = st.file_uploader("Upload beneficiary dataset (CSV)", type=["csv"])

if file:
    st.write("**Step 2:** File uploaded successfully")

    # Read CSV
    df = pd.read_csv(file)
    st.write("Data preview:", df.head())
    
    n_clusters = st.slider("Select Number of Clusters to Train",min_value=2, max_value=10)


    if st.button("Train"):

        # Preprocess data
        data_transformer = DataTransformation()
        preprocessor = data_transformer.get_column_transformer_object()
        st.write("Data Transformation object has been created.")

        #import and combine clustering pipeline with transformation pipeline 

        st.write("**Step 3:** Training KMeans model")

        model_trainer = ModelTrainer(data=df,
                                        preprocessor_object=preprocessor)
        
        
                    
                
            
        
        model_trainer_artifact = model_trainer.train_kmeans(n_clusters= n_clusters)

        st.write("Training complete!")

        st.write(f"**Step 4:** Model and Preprocessor saved successfully in pipeline object at {model_trainer_artifact.model_store_full_path}")

        # Saving model and preprocessor
        df['Cluster'] = model_trainer_artifact.full_pipeline_object.predict(df)

        st.session_state['training_data'] = df 


        st.session_state['pipeline'] = model_trainer_artifact.full_pipeline_object
        st.session_state["transformed_data_path"] = model_trainer_artifact.x_transformed_path
            
        
        