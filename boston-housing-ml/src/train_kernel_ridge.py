from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values
    data = data.fillna(data.mean())
    # Separate features and target variable
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    return X, y

# Train the Kernel Ridge model
def train_kernel_ridge(X_train, y_train):
    model = KernelRidge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

if __name__ == "__main__":
    # Define file path
    filepath = os.path.join('data', 'boston_housing.csv')  # Adjust the path as necessary
    # Load and preprocess data
    data = load_data(filepath)
    X, y = preprocess_data(data)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = train_kernel_ridge(X_train, y_train)
    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    # Save the model
    joblib.dump(model, 'kernel_ridge_model.pkl')