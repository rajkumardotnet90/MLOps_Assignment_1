def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def evaluate_multiple_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        mse = evaluate_model(model, X_test, y_test)
        results[name] = mse
    return results