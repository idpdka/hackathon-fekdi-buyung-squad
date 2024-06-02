import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def format_idr(amount):
    formatted_amount = "{:,.0f}".format(amount)
    formatted_amount = "Rp {}".format(formatted_amount)
    formatted_amount = formatted_amount.replace(",", ".")

    return formatted_amount

def get_marker_color(current_stock, max_stock):
    if current_stock > 0.75 * max_stock:
        return 'red'
    elif current_stock > 0.5 * max_stock:
        return 'orange'
    else:
        return 'green'

def preprocess_inference_data(X_data):
    # Normalize transaction_date
    X_data['transaction_date'] = pd.to_datetime(X_data['transaction_date'])
    X_data['transaction_date'] = X_data['transaction_date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

    # Standardize features
    scaler = StandardScaler()
    minmax_scaler_transaction = MinMaxScaler()
    minmax_scaler_price = MinMaxScaler()

    # Apply standard scaling to length, width, height, weight
    X_data[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.fit_transform(X_data[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])

    # Apply MinMax scaling to transaction_date and historical_price
    X_data[['transaction_date']] = minmax_scaler_transaction.fit_transform(X_data[['transaction_date']])
    X_data[['historical_price']] = minmax_scaler_price.fit_transform(X_data[['historical_price']])

    return X_data, minmax_scaler_price