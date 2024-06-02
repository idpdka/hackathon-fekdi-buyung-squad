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

from utils import format_idr, get_marker_color, preprocess_inference_data, get_prediction
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
        Kami menyediakan beberapa model untuk membantu anda memilih harga lelang terbaik untuk ikan anda, yakni:
    ''')

    models = pd.DataFrame.from_dict([
        {"model_name": "XGBoost", "Accuracy": "90%", "MAE": "90%"},
        {"model_name": "Linear Regression", "Accuracy": "90%", "MAE": "90%"},
    ])

    st.table(models)
    selected_fish = st.selectbox(
        "Silakan memilih jenis ikan yang anda inginkan:",
        fish_product_type
    )
    selected_model = st.selectbox(
        "Silakan memilih model yang anda inginkan:",
        ("XGBoost", "Linear Regression")
    )

    selected_fish_title = selected_fish.title()
    preprocess_df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=False)
    preprocess_df, scaler = preprocess_inference_data(preprocess_df)
    predict_df = preprocess_df[preprocess_df[f"fish_product_type_{selected_fish}"] == True].tail(1)
    prediction = model.predict(predict_df)

    if selected_model == "XGBoost":
        prediction = get_prediction(xgboost_model, df, selected_fish)
    else:
        prediction = get_prediction(linear_regression_model, df, selected_fish) 

    with st.container(border=True):
        recommendation_col = st.columns(1)
        st.metric(f"Ikan {selected_fish_title}, model {selected_model}", f"{format_idr(np.exp(prediction.reshape(1, -1))[0][0])}", "+15.5%")

## Chart
# Plot the line chart
with st.container(border=True):
    fish_timeseries = df[df["fish_product_type"] == selected_fish].tail(12)
    st.header(f"Data Historis Ikan {selected_fish_title}", divider='grey')

    with st.container(border=True):
        fig = px.line(
            x=fish_timeseries['transaction_date'],
            y=fish_timeseries['historical_price']
        )
        st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.header('Lokasi Pelelangan Ikan', divider='grey')

    with st.container(border=True):
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