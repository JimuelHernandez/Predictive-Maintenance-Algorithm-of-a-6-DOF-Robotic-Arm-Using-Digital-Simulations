# Predictive Maintenance Algorithm for ABB IRB 4600

This repository contains code for making predictions on new data using a trained machine learning model to predict structural damage in components based on certain features. The provided code includes preprocessing steps, model loading, prediction generation, evaluation metrics calculation, and result visualization.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [File Descriptions](#file-descriptions)
- [License](#license)


## Introduction

This repository provides a machine learning-based solution for predicting structural damage based on input features such as strain, displacement, stress, life, and thermal properties.

## Usage

To utilize the provided code for making predictions on new data, follow these steps:

1. **Data Preparation**: Ensure that the new dataset is formatted correctly and contains the necessary columns required for prediction.

2. **Load Trained Model and Preprocessing Objects**: Load the trained machine learning model (`trained_model.pkl`), the scaler used for feature scaling (`scaler.pkl`), and the feature selector (`feature_selector.pkl`).

3. **Load New Data**: Use the `pd.read_csv()` function to load the new dataset into a pandas DataFrame.

4. **Make Predictions**: Utilize the trained machine learning models that have been loaded to make predictions on the new dataset. The damage prediction results that were made by the trained model will be saved into a new CSV file for further analysis.


## Dependencies

The code in this repository relies on the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- joblib


## File Descriptions

- **trained_model.pkl:** Pickle file containing the trained machine learning model.

- **scaler.pkl:** Pickle file containing the scaler used for feature scaling.

- **feature_selector.pkl:** Pickle file containing the feature selector.

- **P3-300.csv:** Sample dataset for demonstration purposes.

- **P3-300-prediction.csv:** Output CSV file containing predictions on the new data.


## License

This project is licensed under the MIT License. See the LICENSE file for more details.


