import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import create_target, drop_columns, encode_categorical
from src.data_loader import load_data

# Sample fixture: create a small dummy DataFrame for testing preprocessing functions.
@pytest.fixture
def dummy_data():
    data = pd.DataFrame({
        "Numtppd": [0, 1, 0, 2],
        "Numtpbi": [10, 20, 10, 30],
        "Indtppd": [5, 6, 5, 7],
        "Indtpbi": [8, 9, 8, 10],
        "CalYear": [2020, 2021, 2020, 2021],
        "Gender": ["M", "F", "F", "M"],
        "Type": ["A", "B", "A", "B"],
        "Category": ["X", "Y", "X", "Y"],
        "Occupation": ["O1", "O2", "O1", "O2"],
        "SubGroup2": ["S1", "S2", "S1", "S2"],
        "Group2": ["G1", "G2", "G1", "G2"],
        "Group1": ["GR1", "GR2", "GR1", "GR2"]
    })
    return data

def test_create_target(dummy_data):
    data = create_target(dummy_data.copy(), column="Numtppd")
    assert "target" in data.columns
    # Check that rows with non-zero Numtppd get a target of 1
    assert data.loc[1, "target"] == 1

def test_drop_columns(dummy_data):
    columns_to_drop = ["Numtppd", "Numtpbi"]
    data = drop_columns(dummy_data.copy(), columns_to_drop)
    for col in columns_to_drop:
        assert col not in data.columns

def test_encode_categorical(dummy_data):
    categorical_cols = ['CalYear', 'Gender']
    data = encode_categorical(dummy_data.copy(), categorical_cols)
    # After one-hot encoding, original columns should be removed
    for col in categorical_cols:
        assert col not in data.columns

def test_load_data(tmp_path):
    # Create a dummy CSV file and test if load_data reads it properly.
    csv_path = tmp_path / "dummy.csv"
    df = pd.DataFrame({"col1": [1, 2, 3]})
    df.to_csv(csv_path, index=False)
    loaded_df = load_data(str(csv_path))
    pd.testing.assert_frame_equal(df, loaded_df)
