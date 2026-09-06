import pandas as pd
import numpy as np
import os

def create_time_features(df):
    """Extract hour of day from the elapsed Time column."""
    # Kaggle 'Time' is seconds elapsed since first transaction.
    df['hour_of_day'] = (df['Time'] // 3600) % 24
    return df

def create_amount_features(df):
    """Log transform the Amount column to handle extreme outliers."""
    # log1p safely handles taking the log of 0 (since log(0) is undefined).
    df['amount_log'] = np.log1p(df['Amount'])
    return df

def create_synthetic_identities(df, seed=42):
    """Generate fake Customer and Merchant IDs to simulate real-world behavior."""
    np.random.seed(seed)

    # We will simulate 50,000 unique customers and 10,000 unique merchants
    df['customer_id'] = np.random.randint(1, 50000, size=len(df))
    df['merchant_id'] = np.random.randint(1, 10000, size=len(df))

    return df

def create_velocity_features(df):
    """Create behavioral features based on the synthetic customer histories."""
    # Average amount spent by each customer across the dataset
    df['customer_avg_amount'] = df.groupby('customer_id')['Amount'].transform('mean')

    # Total number of transactions by each customer
    df['customer_txn_count'] = df.groupby('customer_id')['Amount'].transform('count')

    # Flag to see if the current transaction is 3x higher than the customer's average
    # Adding a small epsilon (1.0) to avoid division by zero
    df['high_amount_flag'] = (df['Amount'] > (df['customer_avg_amount'] * 3.0)).astype(int)

    return df

def run_feature_engineering():
    input_file = 'data/raw/creditcard.csv'
    output_file = 'data/processed/processed_creditcard.csv'

    print("1. Loading raw dataset...")
    df = pd.read_csv(input_file)

    print("2. Generating time features...")
    df = create_time_features(df)

    print("3. Generating amount transformations...")
    df = create_amount_features(df)

    print("4. Enriching with synthetic Customer/Merchant IDs...")
    df = create_synthetic_identities(df)

    print("5. Generating velocity & behavioral features...")
    df = create_velocity_features(df)

    print(f"6. Saving featured data to {output_file}...")
    df.to_csv(output_file, index=False)

    print("\nFeature Engineering Complete! ✅")
    print(f"New dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("New Columns Added:")
    print(" - hour_of_day\n - amount_log\n - customer_id\n - merchant_id")
    print(" - customer_avg_amount\n - customer_txn_count\n - high_amount_flag")

if __name__ == "__main__":
    run_feature_engineering()
