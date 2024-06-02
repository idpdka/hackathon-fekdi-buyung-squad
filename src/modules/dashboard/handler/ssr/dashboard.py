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

from utils import format_idr, get_marker_color, preprocess_inference_data, get_prediction, get_historical_data
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
df['month'] = pd.to_datetime(df["transaction_date"]).dt.to_period('M')
df['year'] = pd.to_datetime(df["transaction_date"]).dt.to_period('Y')

cold_storage_positions = pd.read_csv("./position/position.csv")
fish_product_type = list(df["fish_product_type"].unique())
xgboost_model = joblib.load("./ml/model/buyung_squad_xgb_v1.pkl")
linear_regression_model = joblib.load("./ml/model/buyung_squad_lr_v1.pkl") 

# Main Dashboard
## Title and description
st.title(f"Buyung Squad: Lelang Ikan Digital")

## Recommendations
with st.container(border=True):
    st.header(f"Rekomendasi Harga Lelang", divider='grey')
    st.caption('''
        Kami memberikan rekomendasi harga lelang ikan tertentu berdasarkan data historis hasil lelang ikan sebelumnya, dalam periode tertentu.
        Data tersebut kami masukkan ke dalam model _machine learning_ untuk membantu optimisasi dan memberikan rekomendasi harga lelang ikan se-akurat mungkin berdasarkan data yang tersedia.
        Kami menyediakan beberapa model untuk membantu anda memilih harga lelang terbaik untuk ikan anda.
    ''')

    fish_selector_col, model_selector_col = st.columns(2)

    with fish_selector_col:
        selected_fish = st.selectbox(
            "Silakan memilih jenis ikan yang anda inginkan:",
            fish_product_type
        )
    
    with model_selector_col:
        selected_model = st.selectbox(
            "Silakan memilih model yang anda inginkan:",
            ("XGBoost", "Linear Regression")
        )

    selected_fish_title = selected_fish.title()
    fish_timeseries = df[df["fish_product_type"] == selected_fish].tail(12)
    preprocess_df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=False)

    if selected_model == "XGBoost":
        prediction = get_prediction(xgboost_model, df, selected_fish)
    else:
        prediction = get_prediction(linear_regression_model, df, selected_fish) 

    with st.container(border=True):
        recommendation_price = np.exp(prediction.reshape(1, -1))[0][0]
        previous_price = fish_timeseries['historical_price'].tail(1).item()
        price_diff = round((recommendation_price - previous_price) / previous_price * 100, 2)

        st.metric(f"Ikan {selected_fish_title}, model {selected_model}", f"{format_idr(recommendation_price)}", f"{price_diff}%")

## Chart
# Plot the line chart
with st.container(border=True):
    st.header(f"Data Historis Ikan {selected_fish_title}", divider='grey')

    selected_freq= st.selectbox(
        "Frekuensi:",
        ("Harian", "Bulanan")
    )
    fish_timeseries = get_historical_data(df, selected_fish, selected_freq)

    with st.container(border=True):
        fig = px.line(
            fish_timeseries,
            x='transaction_date',
            y='historical_price',
            labels={"transaction_date": "Tanggal", "historical_price": "Harga"}
        )
        st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.header('Lokasi Pelelangan Ikan', divider='grey')

    with st.container(border=True):
        m = folium.Map(location=[-7.4015301, 111.3630878], zoom_start=7.5)

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
        st_folium(m, width=700, height=500)