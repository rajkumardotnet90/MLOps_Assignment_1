# Boston Housing ML Project

This project implements regression models to predict housing prices in Boston using various machine learning techniques. The primary focus is on Kernel Ridge Regression and Decision Tree Regression.

## Project Structure

```
boston-housing-ml
├── src
│   ├── train_kernel_ridge.py      # Implementation of Kernel Ridge regression model
│   ├── evaluate.py                 # Functions for evaluating regression models
│   ├── misc.py                     # Custom functions for data loading, preprocessing, training, and evaluation
│   └── train.py                    # Logic to train a DecisionTreeRegressor model
├── tests
│   └── test_train_kernel_ridge.py  # Unit tests for Kernel Ridge model training and evaluation
├── requirements.txt                # List of necessary packages
└── README.md                       # Project documentation
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd boston-housing-ml
pip install -r requirements.txt
```

## Usage

### Training the Decision Tree Regressor

To train the Decision Tree Regressor model, run the following command:

```bash
python src/train.py
```

This will load the data, preprocess it, train the model, and display the average Mean Squared Error (MSE) score on the test set.

### Running Tests

To run the unit tests for the Kernel Ridge model, use:

```bash
pytest tests/test_train_kernel_ridge.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.