import sys
import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import pairwise_distances
from src.exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.training_pipeline_object = st.session_state["pipeline"]
        self.training_data = st.session_state["training_data"]
        try:
            self.transformed_data = np.load(st.session_state["transformed_data_path"])["X_transformed"]
        except Exception as e:
            self.transformed_data = None
            self.logger.error(f"Error loading transformed data: {e}")
            st.error(f"Error loading transformed data. {e}")
            
            
    def predict(self, X):
        if self.transformed_data is None:
            st.error("No trained model or transformed data available. Please train the model and upload data first.")
            return None
        
        try:
            predictions = self.training_pipeline_object.predict(X)
            return predictions
        
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            st.error(f"Error in prediction. {e}")
            return None
        
    def find_nearby_matches_with_clusters(self,
                                          person_details, 
                                          full_pipeline, 
                                          df_original, 
                                          x_transformed, 
                                          n_matches=5):
        
        # Convert the input person details to a DataFrame (ensure column order matches)
        person_df = pd.DataFrame([person_details], columns=df_original.drop(columns="Cluster").columns)

        # Apply the same preprocessing (including one-hot encoding) to person_details
        person_transformed = full_pipeline['preprocessor'].transform(person_df)

        # Predict the cluster for the new input person
        predicted_cluster = full_pipeline['cluster'].predict(person_transformed)[0]

        # Filter the original DataFrame to include only rows from the same cluster
        cluster_indices = df_original[df_original['Cluster'] == predicted_cluster].index
        x_clustered_transformed = x_transformed[cluster_indices]

        # Calculate distances between the new input and the transformed data within the same cluster
        distances = pairwise_distances(person_transformed, x_clustered_transformed)

        # Get the indices of the closest matches within the cluster
        closest_indices_in_cluster = np.argsort(distances[0])[:n_matches]

        # Retrieve the indices of the original data for those closest matches
        closest_indices = cluster_indices[closest_indices_in_cluster]

        # Get the closest matches from the original DataFrame
        closest_matches_df = df_original.iloc[closest_indices].copy()
        
        # Add a column for the distance
        closest_matches_df['Distance'] = distances[0, closest_indices_in_cluster]

        return closest_matches_df
        
        
    def match(self,
              input_df) -> pd.DataFrame:
        try:
            matches = self.find_nearby_matches_with_clusters(
                person_details=input_df,
                full_pipeline=self.training_pipeline_object,
                df_original=self.training_data,
                x_transformed=self.transformed_data,
                n_matches=5
                
            )
            
            return matches
        
        except Exception as e:
            raise CustomException(e, sys )
            
        
        
        