import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading data...")
    df = pd.read_csv('data/raw/creditcard.csv')

    print("\n--- DATA OVERVIEW ---")
    print(f"Total Transactions: {len(df)}")

    # 1. Check missing values
    missing = df.isnull().sum().max()
    print(f"Max Missing Values in any column: {missing}")

    # 2. Check class imbalance
    # Class 0 = Legitimate, Class 1 = Fraud
    legit = len(df[df['Class'] == 0])
    fraud = len(df[df['Class'] == 1])
    fraud_percent = (fraud / len(df)) * 100

    print("\n--- CLASS IMBALANCE ---")
    print(f"Legitimate (0): {legit}")
    print(f"Fraudulent (1): {fraud}")
    print(f"Fraud Percentage: {fraud_percent:.3f}%")

    if fraud_percent < 1.0:
        print("WARNING: Highly Imbalanced Dataset! Standard Accuracy will be misleading.")

    # 3. Analyze Amounts
    print("\n--- TRANSACTION AMOUNT ($) ---")
    print("Legitimate Transactions:")
    print(df[df['Class'] == 0]['Amount'].describe()[['mean', 'min', 'max']])
    print("\nFraudulent Transactions:")
    print(df[df['Class'] == 1]['Amount'].describe()[['mean', 'min', 'max']])

    # 4. Generate a Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.yscale('log') # Log scale because legitimate is huge compared to fraud
    plt.ylabel('Count (Log Scale)')

    # Save the plot
    output_img = 'notebooks/class_distribution.png'
    plt.savefig(output_img)
    print(f"Visualization saved to: {output_img}")
    print("\nEDA Complete! 🚀")

if __name__ == "__main__":
    perform_eda()
