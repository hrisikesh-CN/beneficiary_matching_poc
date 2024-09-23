import os
import streamlit as st 



from src.components.data_transformation import DataTransformation
from src.components.create_graphs import (plot_kmeans_2d, plot_kmeans_3d, 
                                          get_optimal_k_value_from_elbow_method,
                                          plot_silhouette_scores)
import streamlit as st 
import pandas as pd 
from src.utils import dict_to_markdown_table

st.title("Determine the number of clusters")

# File uploader
df = None
if "uploaded_data" not in st.session_state:
    file = st.file_uploader("Upload beneficiary dataset (CSV)", type=["csv"])
    

    if file:
        # del the previous files in artifact 

        if os.path.exists("artifacts"):
            for file_name in os.listdir("artifacts"):
                os.system(f" rm -rf {os.path.join('artifacts', file_name)}")
        st.write("**Step 2:** File uploaded successfully")

        # Read CSV
        df = pd.read_csv(file)
        
else:
    df = st.session_state["uploaded_data"]
    
if df is not None:
    df = df.sample(n=5000, random_state=40)  #due to memory issues
    st.write("Data preview:", df.head())


    # Preprocess data
    data_transformer = DataTransformation()
    preprocessor = data_transformer.get_column_transformer_object()
        
        

    x_transformed = preprocessor.fit_transform(df)

    # Streamlit app to display the Elbow Curve plot
    st.title('Elbow Method to Determine Optimal k')
    with st.spinner('Creating Elbow Curve Plot'):
        fig, optimal_k = get_optimal_k_value_from_elbow_method(transformed_data=x_transformed)
        st.session_state["optimal_k"] = optimal_k
        st.plotly_chart(fig)



    st.subheader("Check the silhouette scores for different values of k")
    with st.spinner('Creating silhouette scores for different values of k'):
        silhoutte_plot_fig, score_map = plot_silhouette_scores(x_transformed,
                                                    start=2,
                                                    end=5)
        st.plotly_chart(silhoutte_plot_fig)
        
    score_map_markdown = dict_to_markdown_table(data_dict=score_map)    
    st.markdown(
        f"### The optimal value of k (clusters) is {optimal_k} with a silhoutte score of `{score_map.get(optimal_k):.4f}`\n\n")
    
    st.markdown(f"#### Silhoutte Scores\n{score_map_markdown}\n\nYou can see how the cluster numbers are seperating the data in the section below.\n\n")



    st.write("Select a range of clusters")
    input_range = st.number_input("input range", step = 1, min_value =2) 
    output_range = st.number_input("output range", step = 1,
                                    min_value=3)

    graph_name = st.radio("Select graph type",["2d graph","3d graph"])
    
   
    

    if st.button("Get Graph"):

        if graph_name == "3d graph":
            plot_data = df.sample(n=3000, random_state=40)
            
        else:
            plot_data = df.copy()
        

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


