# Boston Housing Price Prediction

This project demonstrates a complete machine learning workflow for predicting house prices using classical models on the Boston Housing dataset.

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## Running the Models

- **Decision Tree Regressor**
  ```
  python train_decision_tree.py
  ```

- **Kernel Ridge Regressor**
  ```
  python train_kernel_ridge.py
  ```

Both scripts will print the Mean Squared Error (MSE) on the test set.

## Dataset

The Boston Housing dataset is loaded automatically using the script in `data/download_boston.py`.

## CI/CD

On every push, GitHub Actions will:
- Install dependencies
- Run both model training scripts
- Display the MSE for each model in the Actions log

Check `.github/workflows/ci.yml` for details.

## Submitted By

Rajkumar Chandrasekaran
- G24AI2078
