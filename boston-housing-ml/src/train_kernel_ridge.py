import sys
import os

# Import custom functions from misc.py
sys.path.append(os.path.join(os.path.dirname(__file__)))
from misc import load_data, preprocess_data, evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
import joblib

def train_kernel_ridge():
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Kernel Ridge model
    model = KernelRidge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

    # Save the model
    joblib.dump(model, 'kernel_ridge_model.pkl')

if __name__ == "__main__":
    train_kernel_ridge()