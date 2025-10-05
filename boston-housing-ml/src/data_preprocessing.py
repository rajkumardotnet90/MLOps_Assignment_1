import sys
import os

# Add the data directory to the path and import load_data from download_boston.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from download_boston import load_data

def preprocess_data(data):
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop('MEDV', axis=1))
    
    # Create a DataFrame with scaled features
    import pandas as pd
    scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
    scaled_data['MEDV'] = data['MEDV'].values  # Add target variable back
    
    return scaled_data

def split_data(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test