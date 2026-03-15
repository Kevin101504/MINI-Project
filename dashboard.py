import streamlit as st
import pandas as pd

st.title("AI Demand Forecasting System")

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")

    st.dataframe(df.head())