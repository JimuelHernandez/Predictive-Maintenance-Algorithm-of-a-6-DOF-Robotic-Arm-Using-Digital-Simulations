import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib

# Load the datasets
data_df = pd.read_csv('P1-100.csv')
data_df2 = pd.read_csv('P1-200.csv')
data_df3 = pd.read_csv('P1-300.csv')

# Concatenate all datasets
combined_data_df = pd.concat([data_df, data_df2, data_df3], ignore_index=True)

# Data Cleaning
if combined_data_df.isnull().sum().sum() > 0:
    combined_data_df.dropna(inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']])
combined_data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']] = scaled_features

# Extract features and target
X_combined = combined_data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']].values
y_combined = combined_data_df['DAMAGE'].values

# Feature Selection using Random Forest
feature_selector_combined = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
feature_selector_combined.fit(X_combined, y_combined)
X_selected = feature_selector_combined.transform(X_combined)

# Model Training
n_estimators = 100
learning_rate = 0.01
max_depth = 9
gb_regressor_combined = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
gb_regressor_combined.fit(X_selected, y_combined)

# Save the trained model
joblib.dump(gb_regressor_combined, 'trained_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the feature selector
joblib.dump(feature_selector_combined, 'feature_selector.pkl')

# Load the saved model
loaded_model = joblib.load('trained_model.pkl')

# Predict on new data
new_data_df = pd.read_csv('P3-300.csv')
new_data_scaled = scaler.transform(new_data_df[['NODE', 'STRAIN', 'DISPLACEMENT', 'STRESS', 'LIFE', 'THERMAL']])
X_new_selected = feature_selector_combined.transform(new_data_scaled)
y_new_pred = loaded_model.predict(X_new_selected)

# Add predicted damage as a new column in the DataFrame
new_data_df['PREDICTED_DAMAGE'] = y_new_pred

# Save the DataFrame
new_data_df.to_csv('predicted_damage_new_data.csv', index=False)

# Plot actual vs predicted damage values for new data
plt.figure(figsize=(8, 6))
plt.scatter(new_data_df['NODE'], y_new_pred, color='orange', label='Predicted', alpha=0.5)
plt.xlabel('Node')
plt.ylabel('Predicted Damage')
plt.title('Predicted Damage Values for New Data')
plt.legend()
plt.grid(True)
plt.show()
