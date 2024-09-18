from dataclasses import dataclass
from typing import Union, List, Dict
from sklearn.pipeline import Pipeline



@dataclass
class FileHandlerArtifact:
    file_storage_dir: str
    
    
@dataclass
class ModelTrainerArtifact:
    full_pipeline_object: Pipeline
    model_store_full_path: str
    x_transformed_path: str
    