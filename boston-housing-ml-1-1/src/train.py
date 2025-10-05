def train_model(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from misc import load_data, preprocess_data

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Mean Squared Error of Linear Regression model: {mse}")

if __name__ == "__main__":
    train_model()