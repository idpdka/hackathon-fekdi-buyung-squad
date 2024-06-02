from shapely.geometry import shape, Point
import geopandas as gpd
import pandas as pd
import numpy as np
import random

geojson_path = 'gadm41_IDN_1.json'
gdf = gpd.read_file(geojson_path)
stock_average_capacity = 75 # in 10 ton
stock_deviation_percentage = 50 
stock_multiplier = 10

def generate_stock(average, deviation_percentage, multiplier):
    max_stock = int(average - (deviation_percentage / 100 * average) + (random.randrange(0, 2*deviation_percentage, 1) / 100 * average)) * multiplier 
    current_stock = int(random.randrange(0, max_stock, 10))

    return current_stock, max_stock


def generate_positions_within_geojson(gdf, n_points):
    points = []
    names = []
    current_stocks = []
    max_stocks = []
    for province in gdf['NAME_1'].unique():
        for i in range(n_points):
            while True:
                minx, miny, maxx, maxy = gdf['geometry'].total_bounds
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if gdf.contains(pnt).any():
                    points.append(pnt)
                    names.append(f"{province} storage #{i+1}")
                    current_stock, max_stock = generate_stock(stock_average_capacity, stock_deviation_percentage, stock_multiplier)
                    current_stocks.append(current_stock)
                    max_stocks.append(max_stock)
                    break
    return names, points, current_stocks, max_stocks

names, points, current_stocks, max_stocks = generate_positions_within_geojson(gdf, 5)

position_df = pd.DataFrame({
    'lat': [point.y for point in points],
    'lon': [point.x for point in points],
    'name': names,
    'current_stock': current_stocks,
    'max_stock': max_stocks,
})

position_df.to_csv("position.csv")