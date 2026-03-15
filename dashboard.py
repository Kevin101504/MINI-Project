import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from hybrid_model import hybrid_prediction
from datetime import timedelta

st.title("AI Demand Forecasting System")

# Load dataset
data = pd.read_csv("data/demand_forecasting_dataset (1).csv")
data['date'] = pd.to_datetime(data['date'])

# Load models
xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
lstm_model = load_model("models/lstm_model.h5", compile=False)

st.header("Product Configuration")

product_id = st.selectbox("Product", sorted(data.product_id.unique()))
category_id = st.selectbox("Category", sorted(data.category_id.unique()))
store_id = st.selectbox("Store", sorted(data.store_id.unique()))

promotion_flag = st.selectbox("Promotion Active", [0,1])
holiday_flag = st.selectbox("Holiday", [0,1])

price = st.number_input("Price", value=float(data.price.mean()))
economic_index = st.number_input("Economic Index", value=float(data.economic_index.mean()))

current_stock = st.number_input("Current Stock Level", min_value=0)

start_date = st.date_input("Forecast Start Date")

# Historical demand for rolling features
product_history = data[
    (data.product_id == product_id) &
    (data.store_id == store_id)
].sort_values("date")

rolling_7 = product_history.target_demand.tail(7).mean()
rolling_30 = product_history.target_demand.tail(30).mean()

st.write("7-day average demand:", round(rolling_7,2))
st.write("30-day average demand:", round(rolling_30,2))

if st.button("Generate 30-Day Forecast"):

    predictions = []
    dates = []

    for i in range(30):

        future_date = start_date + timedelta(days=i)

        day = future_date.day
        month = future_date.month
        year = future_date.year
        dayofweek = future_date.weekday()

        input_data = pd.DataFrame({
            "product_id":[product_id],
            "category_id":[category_id],
            "store_id":[store_id],
            "price":[price],
            "promotion_flag":[promotion_flag],
            "holiday_flag":[holiday_flag],
            "economic_index":[economic_index],
            "day":[day],
            "month":[month],
            "year":[year],
            "dayofweek":[dayofweek],
            "rolling_7":[rolling_7],
            "rolling_30":[rolling_30]
        })

        # XGBoost prediction
        xgb_pred = xgb_model.predict(input_data)

        # LSTM prediction
        scaled = scaler.transform(input_data)
        lstm_input = scaled.reshape((scaled.shape[0],1,scaled.shape[1]))
        lstm_pred = lstm_model.predict(lstm_input).flatten()

        final_pred = hybrid_prediction(lstm_pred, xgb_pred)

        predictions.append(final_pred[0])
        dates.append(future_date)

    forecast_df = pd.DataFrame({
        "date":dates,
        "predicted_demand":predictions
    }).set_index("date")

    st.subheader("30-Day Demand Forecast")

    st.line_chart(forecast_df)
    
    # -------------------------
    # Historical Demand Chart
    # -------------------------

    st.subheader("Historical Demand Trend")

    chart_data = product_history[['date','target_demand']]
    chart_data = chart_data.set_index("date")

    st.line_chart(chart_data)

    # -------------------------
    # Inventory Recommendation
    # -------------------------

    total_forecast = forecast_df["predicted_demand"].sum()

    st.subheader("Inventory Recommendation")

    st.write("Forecast demand next 30 days:", round(total_forecast,2))
    st.write("Current stock:", current_stock)

    if current_stock < total_forecast:

        reorder_qty = total_forecast - current_stock

        st.error(f"⚠ Restock Recommended: {round(reorder_qty,2)} units")

    else:

        surplus = current_stock - total_forecast

        st.success(f"Stock sufficient. Surplus: {round(surplus,2)} units")

    st.dataframe(forecast_df)