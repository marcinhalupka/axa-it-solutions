import pandas as pd

def create_target(data: pd.DataFrame, column: str = 'Numtppd') -> pd.DataFrame:
    data['target'] = data[column].apply(lambda x: 1 if x != 0 else 0)
    return data

def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return data.drop(columns=columns)

def encode_categorical(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    return pd.get_dummies(data, columns=categorical_columns)
