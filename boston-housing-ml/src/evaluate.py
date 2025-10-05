def evaluate_model(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'Mean Squared Error': mse,
        'R-squared': r2
    }