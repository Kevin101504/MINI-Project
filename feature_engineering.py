def create_features(df):

    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek

    return df

def create_lag_features(df):

    df['lag_7'] = df.groupby(['store_id','product_id'])['historical_sales'].shift(7)
    df['lag_30'] = df.groupby(['store_id','product_id'])['historical_sales'].shift(30)

    return df