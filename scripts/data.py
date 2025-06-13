import pandas as pd


def drop_const_columns(df: pd.DataFrame):
    df.drop(columns=[col for col in df.columns if df[col].nunique() == 1], inplace=True)


def get_data():
    data = pd.read_parquet("data/train.parquet")

    drop_const_columns(data)

    return data
