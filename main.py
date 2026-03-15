import pandas as pd
import numpy as np
import joblib

from preprocessing import load_data
from feature_engineering import create_features
from lstm_model import train_lstm
from xgb_model import train_xgb
from hybrid_model import hybrid_prediction
from evaluation import evaluate
from supply_chain import supply_chain_action
from visualization import plot_predictions, plot_feature_importance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def main():

    print("Loading dataset...")
    df = load_data()

    print("Creating time features...")
    df = create_features(df)

    # Features used for prediction
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
        'dayofweek'
    ]

    X = df[features]
    y = df['target_demand']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    # -------------------------
    # Scale for LSTM
    # -------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "models/scaler.pkl")

    # reshape for LSTM
    X_train_lstm = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )

    X_test_lstm = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # -------------------------
    # Train LSTM
    # -------------------------

    print("Training LSTM model...")

    lstm_model = train_lstm(X_train_lstm, y_train)
    
    lstm_model.save("models/lstm_model.h5")

    lstm_pred = lstm_model.predict(X_test_lstm).flatten()

    # -------------------------
    # Train XGBoost
    # -------------------------

    print("Training XGBoost model...")

    xgb_model = train_xgb(X_train, y_train)
    
    joblib.dump(xgb_model, "models/xgb_model.pkl")

    xgb_pred = xgb_model.predict(X_test)

    # -------------------------
    # Hybrid prediction
    # -------------------------

    print("Combining predictions...")

    final_pred = hybrid_prediction(lstm_pred, xgb_pred)

    # -------------------------
    # Evaluate model
    # -------------------------

    print("Evaluating model...")

    evaluate(y_test, final_pred)
    
    # -------------------------
    # Visualization
    # -------------------------
    
    plot_predictions(y_test, final_pred)
    plot_feature_importance(xgb_model, features)

    # -------------------------
    # Supply chain decision
    # -------------------------

    print("Supply chain recommendation:")

    predicted_demand = final_pred[-1]

    current_stock = 100  # example stock level

    supply_chain_action(predicted_demand, current_stock)


if __name__ == "__main__":
    main()