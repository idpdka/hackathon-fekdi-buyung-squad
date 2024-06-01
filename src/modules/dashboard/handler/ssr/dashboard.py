import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def format_idr(amount):
    # Add thousands separator
    formatted_amount = "{:,.0f}".format(amount)
    # Add 'Rp' prefix
    formatted_amount = "Rp {}".format(formatted_amount)
    formatted_amount = formatted_amount.replace(",", ".")

    return formatted_amount

# Data Preparation
## Load Dataset
df = pd.read_csv("./ml/dataset/buyung_squad_dataset_v1.csv")
df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
df = df.groupby(['transaction_date', "fish_product_type"]).agg({
    "length_cm": np.mean,
    "width_cm": np.mean,
    "height_cm": np.mean,
    "weight_kg": np.mean,
    "historical_price": np.mean
})
df.reset_index(inplace=True)
fish_product_type = list(df["fish_product_type"].unique())

model = joblib.load("./ml/model/buyung_squad_xgb_v1.pkl")

## Create cold storage random point
with open("./src/data/cold_storage/provinces.json", 'r') as file:
    provinces = json.load(file)

random_lats = np.array([])
random_lons = np.array([])
random_values = np.array([])

random_radius = 0.3 
num_samples = 20 # per province
for province in provinces:
    # Generate random latitude and longitude values within the ranges
    random_lats = np.concatenate((random_lats, np.random.uniform(province["latitude"] - random_radius, province["latitude"] + random_radius, num_samples)))
    random_lons = np.concatenate((random_lons, np.random.uniform(province["longitude"] - random_radius, province["longitude"] + random_radius, num_samples)))
    random_values = np.concatenate((random_values, [f"{province['name']} {i+1}" for i in range(num_samples)])) 

data_map = pd.DataFrame({
    'lat': random_lats,
    'lon': random_lons,
    'value': random_values
})

# Sidebar for user input
st.sidebar.title("Filter")
selected_option = st.sidebar.selectbox("Jenis Ikan", fish_product_type)
selected_option_title = selected_option.title()

# Filter Data
fish_timeseries = df[df["fish_product_type"] == selected_option].tail(12)

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

print(df['fish_product_type'].unique())
preprocess_df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=False)
preprocess_df, scaler = preprocess_inference_data(preprocess_df)
predict_df = preprocess_df[preprocess_df[f"fish_product_type_{selected_option}"] == True].tail(1)
prediction = model.predict(predict_df)

# Main Dashboard
## Title and description
st.title(f"Lelang Ikan {selected_option_title}")
st.write("""
""")

## Recommendations
st.subheader(f"Rekomendasi Harga Lelang Ikan {selected_option_title}: {format_idr(np.exp(prediction.reshape(1, -1))[0][0])}")

## Chart
# Plot the line chart
st.subheader(f"Data Historis Ikan {selected_option_title}")

fig = px.line(x=fish_timeseries['transaction_date'], y=fish_timeseries['historical_price'])
st.plotly_chart(fig, use_container_width=True)

## Alternative Map with Pydeck
st.subheader("Lokasi Pelelangan Ikan")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=data_map,
    get_position='[lon, lat]',
    get_radius=5000,  # Radius is in meters
    get_color='[200, 30, 0, 160]',
    pickable=True
)
view_state = pdk.ViewState(
    latitude=0.7893,
    longitude=113.9213,
    zoom=3,
    pitch=0
)
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(r)
