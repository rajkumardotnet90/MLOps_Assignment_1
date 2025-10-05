import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Add the data directory to the path and import load_data from download_boston.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from download_boston import load_data as raw_load_data

def load_data():
    # Use your custom loader from download_boston.py
    return raw_load_data()

def preprocess_data(data):
    # Handle missing values
    data = data.fillna(data.mean())
    scaler = StandardScaler()
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse