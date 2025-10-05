import pytest
import numpy as np
from src.train_kernel_ridge import train_kernel_ridge
from src.evaluate import evaluate_model

def test_train_kernel_ridge():
    # Generate synthetic data for testing
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)

    # Train the Kernel Ridge model
    model = train_kernel_ridge(X_train, y_train)

    # Check if the model is trained
    assert model is not None, "Model should not be None after training"

def test_evaluate_kernel_ridge():
    # Generate synthetic data for testing
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20)

    # Train the Kernel Ridge model
    model = train_kernel_ridge(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)

    # Check if MSE is a non-negative value
    assert mse >= 0, "Mean Squared Error should be non-negative"