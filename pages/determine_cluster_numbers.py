import streamlit as st 

st.set_page_config(page_title = "Determine Cluster Number from Graphs")


from src.components.data_transformation import DataTransformation
from src.components.create_graphs import plot_kmeans_2d, plot_kmeans_3d
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

    # Preprocess data
    data_transformer = DataTransformation()
    preprocessor = data_transformer.get_column_transformer_object()
        
        
    fig = None 

    x_transformed = preprocessor.fit_transform(df)
    st.write("Select a range of clusters")
    input_range = st.number_input("input range", step = 1, min_value =2) 
    output_range = st.number_input("output range", step = 1,
                                   min_value=3)

    graph_name = st.radio("Select graph type",["2d graph","3d graph"])

    if st.button("Get Graph"):

        for k in range(
            input_range, 
            output_range+1
        ):
            st.subheader(f'Plotting {graph_name}\nKMeans Clustering with k={k}')

            
            if graph_name == "2d graph":
                fig = plot_kmeans_2d(
                    k = k,
                    X=x_transformed)
                
            else:
                fig = plot_kmeans_3d(
                    k = k,
                    X=x_transformed)
                
            st.plotly_chart(fig)


