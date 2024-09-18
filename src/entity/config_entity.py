from src.constant import *
from dataclasses import dataclass
from datetime import datetime
import os 

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class BaseArtifactConfig:
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

base_artifact_config: BaseArtifactConfig = BaseArtifactConfig()

@dataclass
class FileHandlerConfig:
    artifact_dir: str = base_artifact_config.artifact_dir
    file_storage_dir: str = os.path.join(
        artifact_dir,
        FILE_STORAGE_ARTIFACT_DIR_NAME
    )
    
    
    

@dataclass
class ModelTrainerConfig:
    
    transformed_data_path = os.path.join(
        base_artifact_config.artifact_dir,
        TRANSFORMED_DATA_STORE_DIR_NAME,
        TRANSFORMED_DATA_OBJECT_NAME
        
    )    
    
    model_store_path: str = os.path.join(
        base_artifact_config.artifact_dir,
        MODEL_STORAGE_DIR_NAME,
        PIPELINE_OBJECT_NAME
    )