import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import json

# Data Preparation
## Load Dataset
fish_names = pd.read_csv("./data/fish/fish_names.csv")['fish_name']
fish_names_dropdown = {v: k for k, v in fish_names.to_dict().items()}

## Create cold storage random point
with open("./data/cold_storage/provinces.json", 'r') as file:
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
selected_option = st.sidebar.selectbox("Jenis Ikan", fish_names_dropdown.keys())

# Main Dashboard
## Title and description
st.title(f"Lelang Ikan {fish_names[fish_names_dropdown[selected_option]]}")
st.write("""
""")

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

## Sample data for the chart
data_chart = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': np.random.randint(1, 100, 4)
})

## Chart
st.subheader("Sample Chart")
fig, ax = plt.subplots()
ax.bar(data_chart['Category'], data_chart['Values'])
st.pyplot(fig)

## Recommendations
st.subheader("Recommendations")
if selected_option == "Option 1":
    st.write("Recommendation for Option 1: Consider increasing the values for Category A.")
elif selected_option == "Option 2":
    st.write("Recommendation for Option 2: Monitor the values for Category B.")
else:
    st.write("Recommendation for Option 3: Category C and D are performing well.")
