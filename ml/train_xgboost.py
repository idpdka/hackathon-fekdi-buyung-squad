import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
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

# XGBoost with Hyperparameter Tuning
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

# Save the model
joblib.dump(best_xgb_model, 'model/buyung_squad_xgb_v1.pkl')

y_val_pred_xgb = best_xgb_model.predict(X_val)
print("XGBoost Validation Performance:")
print(f"MAE: {mean_absolute_error(y_val, y_val_pred_xgb):.2f}")
print(f"MSE: {mean_squared_error(y_val, y_val_pred_xgb):.2f}")
print(f"R2: {r2_score(y_val, y_val_pred_xgb):.2f}")

y_test_pred_xgb = best_xgb_model.predict(X_test)
print("XGBoost Test Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred_xgb):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_test_pred_xgb):.2f}")
print(f"R2: {r2_score(y_test, y_test_pred_xgb):.2f}")

