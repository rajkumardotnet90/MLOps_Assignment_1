# Boston Housing Dataset

The Boston Housing dataset is a well-known dataset in the field of machine learning and statistics. It contains information about various attributes of houses in Boston and is commonly used for regression tasks, particularly for predicting house prices.

## Dataset Source

The dataset can be found in the UCI Machine Learning Repository and is also available through the `sklearn.datasets` module in Python. The original dataset was published in the 1978 paper by Harrison and Rubinfeld.

## Dataset Structure

The dataset consists of 506 samples and 14 attributes, which are as follows:

1. **CRIM**: Per capita crime rate by town
2. **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
3. **INDUS**: Proportion of non-retail business acres per town
4. **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX**: Nitric oxides concentration (parts per 10 million)
6. **RM**: Average number of rooms per dwelling
7. **AGE**: Proportion of owner-occupied units built prior to 1940
8. **DIS**: Weighted distances to five Boston employment centers
9. **RAD**: Index of accessibility to radial highways
10. **TAX**: Full-value property tax rate per $10,000
11. **PTRATIO**: Pupil-teacher ratio by town
12. **B**: \(1000(Bk - 0.63)^2\) where Bk is the proportion of Black residents by town
13. **LSTAT**: Percentage of lower status of the population
14. **MEDV**: Median value of owner-occupied homes in $1000s (target variable)

## Usage

This dataset is used in this project to build machine learning models that predict house prices based on the various features provided. The project includes data preprocessing, exploratory analysis, feature engineering, and model training using classical machine learning algorithms.