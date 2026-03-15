import streamlit as st
import pandas as pd
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from preprocessing import load_data
from feature_engineering import create_features
from hybrid_model import hybrid_prediction


st.title("AI Demand Forecasting System")

st.write("Upload a dataset to predict demand")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df = create_features(df)

    features = [
        'product_id',
        'category_id',
        'store_id',
        'price',
        'promotion_flag',
        'holiday_flag',
        'economic_index',
        'day',
        'month',
        'year',
        'dayofweek',
        'rolling_7',
        'rolling_30'
    ]

    X = df[features]

    scaler = joblib.load("models/scaler.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    lstm_model = load_model("models/lstm_model.h5", compile=False)

    X_scaled = scaler.transform(X)

    X_lstm = X_scaled.reshape(
        (X_scaled.shape[0], 1, X_scaled.shape[1])
    )

    lstm_pred = lstm_model.predict(X_lstm).flatten()

    xgb_pred = xgb_model.predict(X)

    final_pred = hybrid_prediction(lstm_pred, xgb_pred)

    df["predicted_demand"] = final_pred

    st.subheader("Forecast Results")

    st.dataframe(df[["date", "predicted_demand"]])

    st.line_chart(df["predicted_demand"])