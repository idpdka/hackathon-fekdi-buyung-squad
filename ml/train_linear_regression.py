import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

np.random.seed(42)  # For reproducibility
dataset_path = 'dataset/buyung_squad_dataset_v1.parquet'

df = pd.read_parquet(dataset_path)
df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=True)

# Log transformation of target
df['log_current_price'] = np.log(df['current_price'])

# Drop the original target
df.drop('current_price', axis=1, inplace=True)

# Normalize transaction_date
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_date'] = df['transaction_date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

# Split Data into Training, Validation, and Test Sets
X = df.drop('log_current_price', axis=1)
y = df['log_current_price']

# First split: 80% training and 20% test+validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: 10% validation and 10% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Apply standard scaling to length, width, height, weight
X_train[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.fit_transform(X_train[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])
X_val[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.transform(X_val[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])
X_test[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.transform(X_test[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])

# Apply MinMax scaling to transaction_date and historical_price
X_train[['transaction_date', 'historical_price']] = minmax_scaler.fit_transform(X_train[['transaction_date', 'historical_price']])
X_val[['transaction_date', 'historical_price']] = minmax_scaler.transform(X_val[['transaction_date', 'historical_price']])
X_test[['transaction_date', 'historical_price']] = minmax_scaler.transform(X_test[['transaction_date', 'historical_price']])

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save the model
joblib.dump(lr_model, 'model/buyung_squad_lr_v1.pkl')

y_val_pred_lr = lr_model.predict(X_val)
print("Linear Regression Validation Performance:")
print(f"MAE: {mean_absolute_error(y_val, y_val_pred_lr):.2f}")
print(f"MSE: {mean_squared_error(y_val, y_val_pred_lr):.2f}")
print(f"R2: {r2_score(y_val, y_val_pred_lr):.2f}")

y_test_pred_lr = lr_model.predict(X_test)
print("Linear Regression Test Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred_lr):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_test_pred_lr):.2f}")
print(f"R2: {r2_score(y_test, y_test_pred_lr):.2f}")

# Example Data
# print("Training set example:")
# print(X_train.head(), y_train.head())

# print("Validation set example:")
# print(X_val.head(), y_val.head())

# print("Test set example:")
# print(X_test.head(), y_test.head())
