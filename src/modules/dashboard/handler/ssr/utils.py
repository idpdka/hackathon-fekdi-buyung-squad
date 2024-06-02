import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import qrcode
from PIL import Image
import io
import random
import string

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

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

def get_prediction(model, df, selected_option):
    preprocess_df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=False)
    preprocess_df, scaler = preprocess_inference_data(preprocess_df)
    predict_df = preprocess_df[preprocess_df[f"fish_product_type_{selected_option}"] == True].tail(1)
    predict_df.drop(columns=["month", "year"], inplace=True)
    prediction = model.predict(predict_df)

    return prediction

def get_historical_data(df, selected_fish, freq='Harian'):
    if freq=='Bulanan':
        historical_data = df[df["fish_product_type"] == selected_fish].groupby(df['month']).agg({
                "historical_price": np.mean
            }).tail(12)
        historical_data.reset_index(inplace=True)
        historical_data.rename(columns={"month":"transaction_date"}, inplace=True)
        historical_data["transaction_date"] = historical_data["transaction_date"].astype('str')
    else:
        historical_data = df[df["fish_product_type"] == selected_fish]
        last_date = historical_data.iloc[-1]['transaction_date']
        historical_data = df[df["fish_product_type"] == selected_fish].groupby(df['transaction_date']).agg({
                "historical_price": np.mean
            }).tail(get_days_in_month(last_date.year, last_date.month))

        historical_data.reset_index(inplace=True)
    
    return historical_data

def generate_random_qr():
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(random_string)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white')
    return img, random_string
