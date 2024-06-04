import re
import json
from shapely.geometry import shape, Point
import geopandas as gpd
import pandas as pd
import random

def add_space_before_uppercase(s):
    # Add a space before any uppercase letter that is not at the beginning of the string
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

def capitalize_each_word(s):
    return ' '.join(word.capitalize() for word in s.split())

def generate_stock(average, deviation_percentage, multiplier):
    max_stock = int(average - (deviation_percentage / 100 * average) + (random.randrange(0, 2*deviation_percentage, 1) / 100 * average)) * multiplier 
    current_stock = int(random.randrange(0, max_stock, 10))

    return current_stock, max_stock

def generate_fish_stock(total_stock, num_products):
    # Generate num_products-1 random break points and sort them
    break_points = sorted(random.sample(range(1, total_stock), num_products - 1))
    
    # Calculate stock values as differences between break points
    stock_values = [break_points[0]] + \
                   [break_points[i] - break_points[i - 1] for i in range(1, num_products - 1)] + \
                   [total_stock - break_points[-1]]
    
    return stock_values

def generate_positions_within_geojson(gdf, n_points):
    points = []
    provinces = gdf['NAME_1'].unique()
    province_fish_stocks = {}
    names = []
    current_stocks = []
    max_stocks = []
    fish_stocks = []
    for province in provinces:
        province = add_space_before_uppercase(province)
        province_stock = 0
        for i in range(n_points):
            while True:
                minx, miny, maxx, maxy = gdf['geometry'].total_bounds
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if gdf.contains(pnt).any():
                    points.append(pnt)
                    names.append(f"{province} cold storage #{i+1}")
                    current_stock, max_stock = generate_stock(stock_average_capacity, stock_deviation_percentage, stock_multiplier)
                    current_stocks.append(current_stock)
                    max_stocks.append(max_stock)
                    province_stock += current_stock
                    break
    

        fish_stocks = generate_fish_stock(province_stock, len(fish_product_type))
        province_fish_stocks[province] = dict(zip(fish_product_type, fish_stocks))

    return names, points, current_stocks, max_stocks, province_fish_stocks

df = pd.read_csv("./ml/dataset/buyung_squad_dataset_v1.csv")
fish_product_type = list(df["fish_product_type"].unique())
for i in range(len(fish_product_type)):
    fish_product_type[i] = capitalize_each_word(fish_product_type[i])

geojson_path = './position/gadm41_IDN_1.json'
gdf = gpd.read_file(geojson_path)
stock_average_capacity = 75 # in 10 ton
stock_deviation_percentage = 50 
stock_multiplier = 10

names, points, current_stocks, max_stocks, province_fish_stocks = generate_positions_within_geojson(gdf, 5)
with open('./position/fish_stock.json', 'w') as json_file:
    json.dump(province_fish_stocks, json_file, indent=4)

position_df = pd.DataFrame({
    'lat': [point.y for point in points],
    'lon': [point.x for point in points],
    'name': names,
    'current_stock': current_stocks,
    'max_stock': max_stocks,
})

position_df.to_csv("./position/position.csv")