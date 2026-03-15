def hybrid_prediction(lstm_pred, rf_pred):

    final_pred = (lstm_pred + rf_pred) / 2

    return final_pred