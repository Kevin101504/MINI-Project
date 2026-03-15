from xgboost import XGBRegressor

def train_xgb(X_train, y_train):

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    return model