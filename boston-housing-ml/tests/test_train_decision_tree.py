import pytest
import pandas as pd
from src.train_decision_tree import train_and_evaluate

def test_train_and_evaluate():
    # Assuming the function returns a dictionary with 'mse' as one of the keys
    results = train_and_evaluate()
    
    assert isinstance(results, dict)
    assert 'mse' in results
    assert results['mse'] >= 0  # Mean Squared Error should be non-negative

def test_data_shape():
    # Test if the data shape is as expected
    data = pd.read_csv('data/boston_housing.csv')  # Adjust the path as necessary
    assert data.shape == (506, 13)  # Example shape, adjust based on actual dataset

def test_feature_importance():
    results = train_and_evaluate()
    assert 'feature_importances' in results
    assert len(results['feature_importances']) == 13  # Adjust based on actual number of features

def test_model_training():
    results = train_and_evaluate()
    assert results['model'] is not None  # Ensure a model is returned after training