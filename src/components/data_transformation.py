import sys
import pandas as pd
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import category_encoders as ce

from src.encoders import NameSplitter, Top80PercentEncoder
from src.exception import CustomException
from src.constant.training_data import *

class DataTransformation:
    def __init__(self):
        self.preprocessor = None        
        
    def get_column_transformer_object(self):
        try:
            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])



            last_name_pipeline = Pipeline(steps=[
                ('name_splitter', NameSplitter(full_name_column=FULL_NAME_COLUMN_NAME,
                                               last_name_column=LAST_NAME_COLUMN_NAME )
                 
                 ),

                ('last_name_encoding', Top80PercentEncoder(column=LAST_NAME_COLUMN_NAME, top_percent=80, other_value=0),)
            ])


            # Top80PercentEncoder for 'Ethnicity'
            ethnicity_pipeline = Pipeline(steps=[
                ('ethnicity_encoding', Top80PercentEncoder(column='Ethnicity', top_percent=80, other_value=0))
            ])

            # Top80PercentEncoder for 'Religion'
            religion_pipeline = Pipeline(steps=[
                ('religion_encoding', Top80PercentEncoder(column='Religion', top_percent=80, other_value=0))
            ])
            
            geo_pipeline = Pipeline(steps=[
                                            ('scaler', StandardScaler()),  # Scale latitude and longitude
                                        ])
            
            
            binary_encoding_pipeline = Pipeline(steps=[
                                                    ('encoder', ce.BinaryEncoder(cols=BINARY_CATEGORICAL_FEATURES))
                                                ])
            
            
            # Combine all pipelines using ColumnTransformer
            self.preprocessor = ColumnTransformer(transformers=[
                ('numerical_pipeline', numerical_pipeline, NUMERICAL_FEATURES),  # Apply to numerical columns
                ('last_name_pipeline', last_name_pipeline, ['Name']),  # Split Name into last_name
                ('ethnicity_pipeline', ethnicity_pipeline, ['Ethnicity']),  # Encode Ethnicity
                ('religion_pipeline', religion_pipeline, ['Religion']),
                ('geo_pipeline', geo_pipeline, GEO_FEATURES),
                ("binary_category_pipeline", binary_encoding_pipeline, BINARY_CATEGORICAL_FEATURES)

            ])
            
            
            return self.preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    # def transform(self):
    #     try:
    #         pipeline = self.get_column_transformer_object()
    #         transformed_data = pipeline.fit_transform(self.data)
    #         return transformed_data

            


    #     except Exception as e:
    #         raise CustomException(e, sys)