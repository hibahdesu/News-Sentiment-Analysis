import pandas as pd

def load_data():
    df = pd.read_csv("data/news.csv")
    return df
