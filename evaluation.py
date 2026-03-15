from sklearn.metrics import mean_absolute_error
import numpy as np

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())

    print("MAE:", mae)
    print("RMSE:", rmse)