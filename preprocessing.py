import pandas as pd

def load_data():

    df = pd.read_csv("data\demand_forecasting_dataset (1).csv")

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date')

    return df