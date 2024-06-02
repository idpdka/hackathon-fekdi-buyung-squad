import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)  # For reproducibility
n_samples = 50_000 

def generate_generic_dataset():
    # Generate Dummy Data with 10,000 rows
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

    return data

def generate_training_dataset(df):
    df = pd.get_dummies(df, columns=['fish_product_type'], drop_first=False)

    # Log transformation of target
    df['log_current_price'] = np.log(df['current_price'])

    # Drop the original target
    df.drop('current_price', axis=1, inplace=True)

    # Normalize transaction_date
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['transaction_date'] = df['transaction_date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

    # Split Data into Training, Validation, and Test Sets
    X = df.drop('log_current_price', axis=1)
    y = df['log_current_price']

    # First split: 80% training and 20% test+validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Second split: 10% validation and 10% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Apply standard scaling to length, width, height, weight
    X_train[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.fit_transform(X_train[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])
    X_val[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.transform(X_val[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])
    X_test[['length_cm', 'width_cm', 'height_cm', 'weight_kg']] = scaler.transform(X_test[['length_cm', 'width_cm', 'height_cm', 'weight_kg']])

    # Apply MinMax scaling to transaction_date and historical_price
    X_train[['historical_price']] = minmax_scaler.fit_transform(X_train[['historical_price']])
    X_val[['historical_price']] = minmax_scaler.transform(X_val[['historical_price']])
    X_test[['historical_price']] = minmax_scaler.transform(X_test[['historical_price']])

    X_train[['transaction_date']] = minmax_scaler.fit_transform(X_train[['transaction_date']])
    X_val[['transaction_date']] = minmax_scaler.transform(X_val[['transaction_date']])
    X_test[['transaction_date']] = minmax_scaler.transform(X_test[['transaction_date']])

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, val, test


df = pd.DataFrame(generate_generic_dataset())
train, val, test = generate_training_dataset(df)

print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Test set size: {len(test)}")

df.to_csv('dataset/buyung_squad_dataset_v1.csv')

train.to_parquet('dataset/buyung_squad_train_v1.parquet')
val.to_parquet('dataset/buyung_squad_val_v1.parquet')
test.to_parquet('dataset/buyung_squad_test_v1.parquet')
