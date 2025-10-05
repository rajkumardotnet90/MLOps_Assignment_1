# Boston Housing Price Prediction

This project aims to predict house prices using classical machine learning models on the Boston Housing dataset. The workflow includes data preprocessing, model training, evaluation, and testing.

## Project Structure

```
boston-housing-ml
├── data
│   └── README.md
├── notebooks
│   ├── exploratory_analysis.ipynb
│   └── feature_engineering.ipynb
├── src
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_decision_tree.py
│   ├── train_kernel_ridge.py
│   ├── evaluate.py
│   └── utils.py
├── tests
│   ├── test_data_preprocessing.py
│   ├── test_train_decision_tree.py
│   └── test_train_kernel_ridge.py
├── .github
│   └── workflows
│       └── ci.yml
├── requirements.txt
├── .gitignore
├── README.md
└── environment.yml
```

## Installation

To set up the project, you can create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Alternatively, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Use the `data_preprocessing.py` script to load and preprocess the dataset.
2. **Model Training**: Train the models using `train_decision_tree.py` for Decision Tree Regressor and `train_kernel_ridge.py` for Kernel Ridge Regression.
3. **Evaluation**: Evaluate the models using the `evaluate.py` script to calculate performance metrics.
4. **Testing**: Run the tests located in the `tests` directory to ensure the functionality of the code.

## Notebooks

- **Exploratory Analysis**: The `exploratory_analysis.ipynb` notebook provides visualizations and insights into the dataset.
- **Feature Engineering**: The `feature_engineering.ipynb` notebook focuses on techniques to enhance model performance through feature engineering.

## Continuous Integration

The project includes a CI pipeline defined in `.github/workflows/ci.yml`, which automates the testing and validation of the code on every push to the repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.