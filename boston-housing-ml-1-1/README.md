# Boston Housing ML Project

This project aims to predict housing prices in Boston using machine learning techniques. It includes implementations of different models, including a Kernel Ridge regression model.

## Project Structure

```
boston-housing-ml-1
├── src
│   ├── train.py          # Implementation for training a model using a different algorithm
│   ├── train2.py         # Implementation for training a KernelRidge model
│   ├── misc.py           # Custom functions for data loading, preprocessing, and evaluation
│   └── train_kernel_ridge.py # Function to train the KernelRidge model
├── .github
│   └── workflows
│       └── kernelridge.yml # GitHub Actions workflow for CI/CD
├── requirements.txt       # List of dependencies required for the project
└── README.md              # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd boston-housing-ml-1
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To train the initial model, run:
  ```
  python src/train.py
  ```

- To train the Kernel Ridge model, run:
  ```
  python src/train2.py
  ```

## Evaluation

Both models will output their performance metrics, including Mean Squared Error (MSE) on the test dataset.

## GitHub Actions

The project includes a GitHub Actions workflow that triggers on any push to the `kernelridge` branch. It checks out the code, installs dependencies, and runs both `train.py` and `train2.py` to display their performance.

## Models

- **Kernel Ridge Regression**: A regression technique that combines Ridge regression with the kernel trick, allowing it to learn complex relationships in the data.

This project serves as a practical example of applying machine learning techniques to real-world datasets and demonstrates the use of CI/CD practices in software development.