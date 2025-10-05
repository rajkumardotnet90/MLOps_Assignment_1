def train_kernel_ridge(X_train, y_train, alpha=1.0, kernel='linear'):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import mean_squared_error
    from misc import load_data, preprocess_data, evaluate_model

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)

    # Train the Kernel Ridge model
    model = KernelRidge(alpha=alpha, kernel=kernel)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Kernel Ridge Model MSE: {mse}")

    return model