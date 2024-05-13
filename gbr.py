import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# Load the dataset
data_df = pd.read_csv('P1-100.csv')

# Data Cleaning
# Check for missing values
if data_df.isnull().sum().sum() > 0:
    data_df.dropna(inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']])
data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']] = scaled_features

# Extract features (independent variables) and target (dependent variable)
X = data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']].values
y = data_df['DAMAGE'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection using Random Forest
feature_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
feature_selector.fit(X_train, y_train)
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

# Choose hyperparameters for Gradient Boosting Regression
n_estimators = 500  
learning_rate = 0.01  
max_depth = 9  

# Create Gradient Boosting regressor with reduced features
gb_regressor_selected = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
gb_regressor_selected.fit(X_train_selected, y_train)

# Predict damage on the training and testing sets with reduced features
y_train_pred_selected = gb_regressor_selected.predict(X_train_selected)
y_test_pred_selected = gb_regressor_selected.predict(X_test_selected)

# Evaluate the model with reduced features
mse_train_selected = mean_squared_error(y_train, y_train_pred_selected)
mae_train_selected = mean_absolute_error(y_train, y_train_pred_selected)
rmse_train_selected = np.sqrt(mse_train_selected)
r2_train_selected = r2_score(y_train, y_train_pred_selected)

mse_test_selected = mean_squared_error(y_test, y_test_pred_selected)
mae_test_selected = mean_absolute_error(y_test, y_test_pred_selected)
rmse_test_selected = np.sqrt(mse_test_selected)
r2_test_selected = r2_score(y_test, y_test_pred_selected)

print("Evaluation Metrics on Training Set with Reduced Features:")
print("Mean Squared Error (MSE):", mse_train_selected)
print("Mean Absolute Error (MAE):", mae_train_selected)
print("Root Mean Squared Error (RMSE):", rmse_train_selected)
print("R-squared (R2) Score:", r2_train_selected)

print("\nEvaluation Metrics on Testing Set with Reduced Features:")
print("Mean Squared Error (MSE):", mse_test_selected)
print("Mean Absolute Error (MAE):", mae_test_selected)
print("Root Mean Squared Error (RMSE):", rmse_test_selected)
print("R-squared (R2) Score:", r2_test_selected)

# Plot actual vs predicted damage values for training set with reduced features
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred_selected, color='blue', label='Actual vs Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Damage (Training)')
plt.ylabel('Predicted Damage (Training)')
plt.title('Actual vs Predicted Damage Values (Training Set)')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs predicted damage values for testing set with reduced features
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_selected, color='green', label='Actual vs Predicted (Testing)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Damage (Testing)')
plt.ylabel('Predicted Damage (Testing)')
plt.title('Actual vs Predicted Damage Values (Testing Set)')
plt.legend()
plt.grid(True)
plt.show()

# Predict damage values using the selected features
X_pred_selected = feature_selector.transform(X)
y_pred_selected = gb_regressor_selected.predict(X_pred_selected)

# Revert scaled features back to original scale
original_features = scaler.inverse_transform(X)
data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']] = original_features

# Add predicted damage as a new column in the DataFrame
data_df['PREDICTED_DAMAGE'] = np.concatenate([y_pred_selected])

# Save the DataFrame
data_df.to_csv('gbrresults.csv', index=False)

# Plot actual vs predicted damage values with NODE for both sets
plt.figure(figsize=(8, 6))
plt.scatter(data_df['NODE'], data_df['DAMAGE'], color='blue', label='Actual', alpha=0.5)
plt.scatter(data_df['NODE'], y_pred_selected, color='orange', label='Predicted', alpha=0.5)
plt.xlabel('Node')
plt.ylabel('Damage')
plt.title('Actual vs Predicted Damage Values')
plt.legend()
plt.grid(True)
plt.show()

