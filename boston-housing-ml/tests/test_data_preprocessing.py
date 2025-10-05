import pytest
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data

def test_load_data():
    # Test if the data is loaded correctly
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'CRIM' in data.columns  # Check if a known column exists

def test_preprocess_data():
    # Test if preprocessing handles missing values and scales features
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Check for missing values
    assert processed_data.isnull().sum().sum() == 0  # No missing values should remain
    
    # Check if the features are scaled (mean should be close to 0)
    assert abs(processed_data.mean().mean()) < 1e-1  # Mean of features should be close to 0
    assert abs(processed_data.std().mean() - 1) < 1e-1  # Standard deviation should be close to 1