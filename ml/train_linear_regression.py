import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

np.random.seed(42)  # For reproducibility

train = pd.read_parquet('dataset/buyung_squad_train_v1.parquet')
val = pd.read_parquet('dataset/buyung_squad_val_v1.parquet')
test = pd.read_parquet('dataset/buyung_squad_test_v1.parquet')

# Example Data
X_train = train.drop('log_current_price', axis=1)
y_train = train['log_current_price']
print("Training set example:")
print(X_train.head(), y_train.head())

X_val = val.drop('log_current_price', axis=1)
y_val = val['log_current_price']
print("Validation set example:")
print(X_val.head(), y_val.head())

X_test = test.drop('log_current_price', axis=1)
y_test = test['log_current_price']
print("Test set example:")
print(X_test.head(), y_test.head())

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
