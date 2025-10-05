import sys
import os

# Import custom functions from misc.py
sys.path.append(os.path.join(os.path.dirname(__file__)))
from misc import load_data, preprocess_data, evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def train_decision_tree():
    # Load and preprocess the data (no scaling for DecisionTree)
    data = load_data()
    X, y = preprocess_data(data, scale=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    train_decision_tree()