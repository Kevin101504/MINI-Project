import pandas as pd

def create_features(df):

    df['date'] = pd.to_datetime(df['date'])

    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek

    # Rolling demand trends
    df["rolling_7"] = df["target_demand"].rolling(7).mean()
    df["rolling_30"] = df["target_demand"].rolling(30).mean()

    df = df.fillna(method="bfill")

    return df

def create_lag_features(df):

    df['lag_7'] = df.groupby(['store_id','product_id'])['historical_sales'].shift(7)
    df['lag_30'] = df.groupby(['store_id','product_id'])['historical_sales'].shift(30)

    return df
