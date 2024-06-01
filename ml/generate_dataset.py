import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

# Generate Dummy Data with 10,000 rows
n_samples = 1_000 
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_samples).to_numpy()
np.random.shuffle(date_range)
fish_type_and_market_price = {
    'bandeng': 23_000,
    'kerapu': 55_000,
    'kakap': 50_000,
    'udang': 82_000,
    'tuna': 50_000,
}

data = {
    'transaction_date': date_range,
    'fish_product_type': np.random.choice(list(fish_type_and_market_price.keys()), n_samples),
    'length_cm': np.random.uniform(20, 100, n_samples),
    'width_cm': np.random.uniform(5, 25, n_samples),
    'height_cm': np.random.uniform(3, 15, n_samples),
    'weight_kg': 1,
}

# We get auction price bar at random:
# Min price = 20% lower from market price
# Max price = 10% lower from market price

def get_min_price(price):
    return price - (20 * price / 100)

def get_max_price(price):
    return price - (10 * price / 100)

data['historical_price'] = np.array([
    np.random.uniform(
        get_min_price(fish_type_and_market_price[fish]),
        get_max_price(fish_type_and_market_price[fish])
    )
    for fish in data['fish_product_type']
])
data['current_price'] = np.array([
    np.random.uniform(
        get_min_price(fish_type_and_market_price[fish]),
        get_max_price(fish_type_and_market_price[fish])
    )
    for fish in data['fish_product_type']
])

df = pd.DataFrame(data)
df.to_parquet('dataset/buyung_squad_dataset_v1.parquet')
df.to_csv('dataset/buyung_squad_dataset_v1.csv')