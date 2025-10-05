def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop('MEDV', axis=1))
    
    # Create a DataFrame with scaled features
    scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
    scaled_data['MEDV'] = data['MEDV'].values  # Add target variable back
    
    return scaled_data

def split_data(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test