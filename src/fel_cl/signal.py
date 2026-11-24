import pandas as pd


def rolling_mean(data, window_size=50):
    series = pd.Series(data)
    return series.rolling(window=window_size).mean().to_numpy()
