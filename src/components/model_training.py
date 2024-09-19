import os, sys
import pandas as pd 

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact

from src.logger import get_logger
from src.exception import CustomException
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline
import joblib


class ModelTrainer:
    def __init__(self,
                 data:pd.DataFrame,
                 preprocessor_object):
        
        self.data = data
        self.preprocessor_object = preprocessor_object
        self.model_trainer_config = ModelTrainerConfig()
        self.logger = get_logger(__name__)
        
        
        

        
   
        
        
    def train_kmeans(self,
                      n_clusters):
        
        try:
            # Define the full pipeline including clustering
            full_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor_object),
                ('cluster', KMeans(n_clusters=n_clusters))  
            ])
            
            self.x_transformed = self.preprocessor_object.fit_transform(self.data)
            
            # Fit the full pipeline
            full_pipeline.fit(self.data)
            
            pipeline_store_path = self.model_trainer_config.model_store_path
            os.makedirs(
                    os.path.dirname(pipeline_store_path)
                    , exist_ok=True
            )
            
            # Save the trained model
            joblib.dump(full_pipeline,
                        pipeline_store_path
                        )
            
            #save the transformed data 
            transformed_data_store_path = self.model_trainer_config.transformed_data_path
            
            os.makedirs(os.path.dirname(transformed_data_store_path),
                exist_ok=True
            )
            np.savez(
                transformed_data_store_path,
                     X_transformed=self.x_transformed)

            
            model_trainer_artifact = ModelTrainerArtifact(
                full_pipeline_object=full_pipeline,
                model_store_full_path=pipeline_store_path,
                x_transformed_path = transformed_data_store_path
            )
            
            self.logger.info(f'Pipeline object saved at {pipeline_store_path}')

            return model_trainer_artifact
            
                            
        except Exception as e:
            raise CustomException(e, sys)