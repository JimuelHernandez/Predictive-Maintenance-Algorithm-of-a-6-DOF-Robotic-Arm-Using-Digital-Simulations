import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the new dataset
new_data_df = pd.read_csv('P3-300.csv')

# Load the trained model and scaler
loaded_model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_selector_combined = joblib.load('feature_selector.pkl')

# Ensure the same columns are present as in the training data
expected_columns = ['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']
for column in expected_columns:
    if column not in new_data_df.columns:
        raise ValueError(f"Column '{column}' is missing in the new dataset.")

# Preprocessing steps
new_data_scaled = scaler.transform(new_data_df[expected_columns])
X_new_selected = feature_selector_combined.transform(new_data_scaled)

# Make predictions on the new data
y_new_pred = loaded_model.predict(X_new_selected)

# Add predicted damage as a new column 
new_data_df['PREDICTED_DAMAGE'] = y_new_pred

# Calculate evaluation metrics using actual values
y_actual = new_data_df['DAMAGE'].values
mae = mean_absolute_error(y_actual, y_new_pred)
mse = mean_squared_error(y_actual, y_new_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_new_pred)

print("Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

# Save the DataFrame
new_data_df.to_csv('P3-300-prediction.csv', index=False)

# Plot predicted damage along the node
plt.figure(figsize=(8, 6))
plt.scatter(new_data_df['NODE'], new_data_df['DAMAGE'], color='blue', label='Actual Damage', alpha=0.5)
plt.scatter(new_data_df['NODE'], y_new_pred, color='orange', label='Predicted Damage', alpha=0.5)
plt.xlabel('Node')
plt.ylabel('Damage')
plt.title('Actual vs Predicted Damage along Node')
plt.legend()
plt.grid(True)
plt.show()
