import streamlit as st
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go

from utils import format_idr, get_marker_color, preprocess_inference_data
from streamlit_folium import st_folium

# Load Model
model = joblib.load("./ml/model/buyung_squad_xgb_v1.pkl")

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

cold_storage_positions = pd.read_csv("./position/position.csv")

fish_product_type = list(df["fish_product_type"].unique())

# Sidebar for user input
st.sidebar.title("Filter")
selected_option = st.sidebar.selectbox("Jenis Ikan", fish_product_type)
selected_option_title = selected_option.title()

# Filter Data
fish_timeseries = df[df["fish_product_type"] == selected_option].tail(12)

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
# layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=cold_storage_positions,
#     get_position='[lon, lat]',
#     get_radius=5000,  # Radius is in meters
#     get_color='[200, 30, 0, 160]',
#     pickable=True
# )
# view_state = pdk.ViewState(
#     latitude=0.7893,
#     longitude=113.9213,
#     zoom=3,
#     pitch=0
# )
# r = pdk.Deck(layers=[layer], initial_view_state=view_state)
# st.pydeck_chart(r)
# Create a Folium map centered around Indonesia
m = folium.Map(location=[-2.548926, 118.0148634], zoom_start=5)

# Add markers to the map
for _, row in cold_storage_positions.iterrows():
    color = get_marker_color(row['current_stock'], row['max_stock'])
    popup_content = f"""
    <div style="width: 100px;">
        <b>{row['name']}</b><br>
        Current Stock: {row['current_stock']}<br>
        Max Stock: {row['max_stock']}
    </div>
    """
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(popup_content, max_width=250),
        icon=folium.Icon(color=color)
    ).add_to(m)

# Display the map in Streamlit
st.title("Warehouse Stock Levels in Aceh")
st_folium(m, width=700, height=500)