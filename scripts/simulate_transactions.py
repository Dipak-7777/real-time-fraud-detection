"""
Real-time transaction simulator for fraud detection system.
Generates and sends synthetic transactions to the API.
"""
import requests
import random
import time
import numpy as np
from datetime import datetime

API_URL = "http://localhost:8000/predict"

def generate_transaction(transaction_num):
    """
    Generate a synthetic transaction.

    80% are normal, 20% are suspicious (to see the model in action)
    """
    # Decide if this will be suspicious or normal
    is_suspicious = random.random() < 0.2

    # Use timestamp to make transaction ID unique across multiple runs
    timestamp_ms = int(time.time() * 1000)
    transaction_id = f"TXN{timestamp_ms}_{transaction_num:04d}"

    if is_suspicious:
        # EXTREMELY suspicious transaction patterns (to trigger 95% threshold)
        amount = random.uniform(2000, 10000)  # VERY high amounts
        hour = random.choice([2, 3, 4])  # Unusual hours (2-4 AM)

        # Generate EXTREMELY outlier V values (model trained on -5 to +5 range)
        v_features = {}
        for i in range(1, 29):
            # Make some V features extremely anomalous
            if i in [4, 11, 14]:  # These are usually important fraud indicators
                v_features[f"V{i}"] = random.uniform(-8, 8)  # Extreme outliers
            else:
                v_features[f"V{i}"] = random.uniform(-4, 4)

        customer_avg = random.uniform(50, 150)  # Low average
        high_amount_flag = 1  # This is unusually high for the customer

    else:
        # Normal transaction patterns
        amount = random.uniform(10, 300)  # Normal amounts
        hour = random.randint(8, 22)  # Normal business hours

        # Generate typical V values (closer to 0)
        v_features = {f"V{i}": random.uniform(-1, 1) for i in range(1, 29)}

        customer_avg = random.uniform(80, 200)
        high_amount_flag = 0

    # Build transaction
    transaction = {
        "transaction_id": transaction_id,
        "Amount": round(amount, 2),
        **v_features,
        "hour_of_day": hour,
        "amount_log": round(np.log1p(amount), 4),
        "customer_avg_amount": round(customer_avg, 2),
        "customer_txn_count": random.randint(5, 100),
        "high_amount_flag": high_amount_flag
    }

    return transaction, is_suspicious

def send_transaction(transaction):
    """Send transaction to the API and return the response."""
    try:
        response = requests.post(API_URL, json=transaction, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}

def run_simulator(num_transactions=20, delay=1.0):
    """
    Run the transaction simulator.

    Args:
        num_transactions: Number of transactions to generate (default 20)
        delay: Seconds between transactions (default 1.0)
    """
    print("="*70)
    print("  REAL-TIME FRAUD DETECTION SIMULATOR")
    print("="*70)
    print(f"Sending {num_transactions} transactions to {API_URL}")
    print(f"Delay: {delay}s between transactions\n")
    print("Legend: 🟢 = Legitimate  |  🔴 = Fraud Detected  |  ⚠️  = Suspicious Pattern\n")

    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code != 200:
            print("❌ API is not responding properly!")
            return
    except:
        print("❌ Cannot connect to API!")
        print("Please start the API first with:")
        print("  uv run uvicorn api.main:app --reload")
        return

    print("✅ Connected to API. Starting simulation...\n")
    print("-"*70)

    fraud_detected = 0
    legitimate = 0

    for i in range(1, num_transactions + 1):
        # Generate transaction
        transaction, was_suspicious_pattern = generate_transaction(i)

        # Send to API
        result = send_transaction(transaction)

        if "error" in result:
            print(f"❌ {transaction['transaction_id']}: {result['error']}")
            continue

        # Display result
        prediction = result["prediction"]
        probability = result["fraud_probability"]
        risk = result["risk_level"]
        latency = result["latency_ms"]

        if prediction == "FRAUD":
            icon = "🔴"
            fraud_detected += 1
        else:
            icon = "🟢"
            legitimate += 1

        # Show if we intentionally made it suspicious
        pattern = "⚠️  Suspicious" if was_suspicious_pattern else "Normal"

        print(f"{icon} {transaction['transaction_id']} | "
              f"${transaction['Amount']:>7.2f} | "
              f"Pred: {prediction:>10} | "
              f"Prob: {probability:.3f} | "
              f"Risk: {risk:>8} | "
              f"{latency:.1f}ms | "
              f"{pattern}")

        # Wait before next transaction
        time.sleep(delay)

    # Summary
    print("-"*70)
    print(f"\n{'SIMULATION COMPLETE':^70}")
    print(f"\nTotal Transactions: {num_transactions}")
    print(f"  🟢 Legitimate:     {legitimate}")
    print(f"  🔴 Fraud Detected: {fraud_detected}")
    print(f"  Fraud Rate:        {(fraud_detected/num_transactions)*100:.1f}%")
    print("\nAll predictions are now stored in the database!")
    print("View them at: http://localhost:8000/transactions")
    print("="*70)

if __name__ == "__main__":
    # You can customize these values
    run_simulator(
        num_transactions=20,  # How many transactions to send
        delay=1.0             # Seconds between each one
    )
