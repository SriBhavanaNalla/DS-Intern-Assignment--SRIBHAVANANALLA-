# Energy Consumption Prediction - Full Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv("data.csv")

# Convert columns to numeric
cols_to_convert = df.columns.drop('timestamp')
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Handle missing values with KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = df.copy()
df_imputed[cols_to_convert] = imputer.fit_transform(df[cols_to_convert])

# Feature Engineering
# Extract date-time features
if 'timestamp' in df.columns:
    df_imputed['timestamp'] = pd.to_datetime(df['timestamp'])
    df_imputed['hour'] = df_imputed['timestamp'].dt.hour
    df_imputed['dayofweek'] = df_imputed['timestamp'].dt.dayofweek
    df_imputed['month'] = df_imputed['timestamp'].dt.month

# Drop timestamp for modeling
df_model = df_imputed.drop(columns=['timestamp'])

# Split features and target
X = df_model.drop(columns='equipment_energy_consumption')
y = df_model['equipment_energy_consumption']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training - Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.4f}")

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
important_features = importances.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
important_features.head(15).plot(kind='barh')
plt.title("Top 15 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Recommendation output
print("\nKey Recommendations:")
print("1. Focus on optimizing conditions in zones with high temperature/humidity impact.")
print("2. Monitor lighting and equipment energy jointly – lighting usage has high correlation.")
print("3. Random variables' importance: \n", importances[['random_variable1', 'random_variable2']])
print("   Consider removing if low importance.")
