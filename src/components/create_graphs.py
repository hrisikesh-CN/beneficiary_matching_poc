import os, sys
import pandas as pd
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact

from src.exception import CustomException
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import silhouette_score
import streamlit as st

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
                        title=f'KMeans Clustering with k={k} (Silhouette Score: {sil_score:.4f})',
                        color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"] )

        # Add centroids
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1],
                                mode='markers',
                                marker=dict(color='Black', size=20, symbol='x'),
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
    
    
    
def get_optimal_k_value_from_elbow_method(transformed_data):
    try:
        inertia = []
        k_values = range(1, 11)

        # Calculate inertia for k=1 to k=10
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(transformed_data)
            inertia.append(kmeans.inertia_)

        # Find the optimal k using KneeLocator
        kneedle = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        # Plotting the Elbow Curve using Plotly
        fig = go.Figure()

        # Add the inertia vs k plot
        fig.add_trace(go.Scatter(x=list(k_values), y=inertia, mode='lines+markers', name='Inertia'))

        # Highlight the optimal k
        if optimal_k is not None:
            fig.add_trace(go.Scatter(
                x=[optimal_k], y=[inertia[optimal_k - 1]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name=f'Optimal k={optimal_k}'
            ))

        # Update plot layout
        fig.update_layout(
            title='Elbow Method to Determine Optimal k',
            xaxis_title='Number of clusters (k)',
            yaxis_title='Inertia',
            showlegend=True
        )

        return fig, optimal_k

    except Exception as e:
        raise Exception(f"Error occurred: {e}")
    
    
    


# Function to calculate silhouette scores and plot the results using Plotly
def plot_silhouette_scores(X_transformed,
                           start:int,
                           end:int):
    sil_scores = []
    list_of_k = [i for i in range(start,end)]

    # Try different values of k (from 2 to 10)
    for k in list_of_k:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_transformed)
        labels = kmeans.labels_
        sil_score = silhouette_score(X_transformed, labels)
        sil_scores.append(sil_score)

    # Create a Plotly line chart for silhouette scores
    fig = go.Figure()

    # Add the silhouette scores vs k plot
    fig.add_trace(go.Scatter(x=list(range(2, 11)), y=sil_scores, mode='lines+markers', name='Silhouette Score'))

    # Update plot layout
    fig.update_layout(
        title='Silhouette Score for Different k Values',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        showlegend=True
    )
    score_map = {k:score for k, score in zip(list_of_k, sil_scores)}

    return fig, score_map