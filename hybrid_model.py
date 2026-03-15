def hybrid_prediction(lstm_pred, xgb_pred):

    final_pred = 0.3 * lstm_pred + 0.7 * xgb_pred

    return final_pred