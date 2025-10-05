from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from src.misc import load_data, preprocess_data

def main():
    # Load and preprocess the data
    X, y = load_data('path/to/your/dataset.csv')  # Update with actual dataset path
    X = preprocess_data(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree Regressor model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Average Mean Squared Error on the test set: {mse:.4f}')

if __name__ == "__main__":
    main()