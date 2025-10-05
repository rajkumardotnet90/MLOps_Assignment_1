def load_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load the dataset (assuming a CSV file)
    data = pd.read_csv('path/to/boston_housing.csv')
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model