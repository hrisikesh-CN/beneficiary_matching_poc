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


def plot_kmeans_2d(k,
                    X):
    
    """
    Plot KMeans clustering for 2D data
    
    Args:
        k (int): Number of clusters
        X (np.ndarray): Transformed 2D data
    
    Returns:
        plotly.graph_objs.Figure: Plot of KMeans clustering
    """
    
    try:
        # Apply PCA to reduce from 9D to 2D for visualization
        pca = PCA(n_components=2)
        X_2D = pca.fit_transform(X)  # X is your 9D data
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_2D)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sil_score = silhouette_score(X_2D, labels)
        
        

        x=X_2D[:, 0]
        y=X_2D[:, 1]
        # Create a scatter plot using Plotly
        fig = px.scatter(x=x, y=y, color=labels.astype(str),
                        labels={'x': 'Feature 1', 'y': 'Feature 2'},
                        title=f'KMeans Clustering with k={k} (Silhouette Score: {sil_score:.4f})')

        # Add centroids
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1],
                                mode='markers',
                                marker=dict(color='red', size=12, symbol='x'),
                                name='Centroids'))

        return fig

    except Exception as e:
        raise CustomException(e, sys)
    
    
def plot_kmeans_3d(k, X):
    """
    Plot KMeans clustering for 3D data
            
    Args:
        k (int): Number of clusters
        X (np.ndarray): Transformed 2D data
    
    Returns:
        plotly.graph_objs.Figure: Plot of KMeans clustering
    """
    
    try:
        
        pca = PCA(n_components=3)
        X_3D = pca.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_3D)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sil_score = silhouette_score(X_3D, labels)
        
        
    

        # Create a 3D scatter plot using Plotly
        fig = go.Figure()
          # X is your 9D data

        x=X_3D[:, 0]
        y=X_3D[:, 1]
        z = X_3D[:, 2]
        # Create a 3D scatter plot using Plotly
        fig = go.Figure()

        # Add data points
        fig.add_trace(go.Scatter3d(
            x=x, 
            y= y,
            z= z,
            mode='markers',
            marker=dict(
                size=5,
                color=labels,  # Color by cluster
                colorscale='Viridis',  # Cluster color scale
                opacity=0.8
            )
        ))

        # Add centroids
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='x',
                opacity=0.9
            ),
            name='Centroids'
        ))

        # Set plot title and labels
        fig.update_layout(
            title=f'KMeans Clustering with k={k} (Silhouette Score: {sil_score:.4f})',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            )
        )

        return fig
    
    except Exception as e:
        raise CustomException(e, sys)