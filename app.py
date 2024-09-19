import streamlit as st 

st.header("Benificiary Matching POC")

upload_data_and_train_page = st.Page(
    "streamlit_pages/train.py", 
    title="Training",
    icon=":material/dashboard:", default=True
)

graph_page = st.Page(
    "streamlit_pages/determine_cluster_numbers.py", 
    icon="📉",
    title="Graphs to Determine Optimal Cluster Numbers",
   default=False
)

predictions_page = st.Page(
    "streamlit_pages/prediction.py", 
    icon="🔼",
    title="Prediction",
    default=False)


pg = st.navigation(
        {
            "Training": [upload_data_and_train_page],
            "Reports": [graph_page],
            "Prediction": [predictions_page],
        }
    )

pg.run()