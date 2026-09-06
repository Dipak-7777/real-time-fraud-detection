"""
Simple API tester for the Fraud Detection API
Run this while the API server is running in another terminal
"""
import requests
import json
import sys

API_BASE = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Root Endpoint")
    print("="*60)
    response = requests.get(f"{API_BASE}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Health Check")
    print("="*60)
    response = requests.get(f"{API_BASE}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Model Info")
    print("="*60)
    response = requests.get(f"{API_BASE}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction():
    """Test the prediction endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Fraud Prediction")
    print("="*60)

    # Load the test transaction
    with open('test_transaction.json', 'r') as f:
        transaction = json.load(f)

    print(f"Sending transaction: {transaction['transaction_id']}")
    print(f"Amount: ${transaction['Amount']}")

    response = requests.post(f"{API_BASE}/predict", json=transaction)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  Transaction ID: {result['transaction_id']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Model Version: {result['model_version']}")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    print("\n🚀 FRAUD DETECTION API TEST SUITE")
    print("="*60)
    print("Make sure the API is running at http://localhost:8000")
    print("(Run 'uv run uvicorn api.main:app --reload' in another terminal)")

    # Check if server is running
    try:
        requests.get(API_BASE, timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to the API server!")
        print("Please start the server first with:")
        print("  uv run uvicorn api.main:app --reload")
        sys.exit(1)

    # Run all tests
    results = []
    results.append(("Root", test_root()))
    results.append(("Health", test_health()))
    results.append(("Model Info", test_model_info()))
    results.append(("Prediction", test_prediction()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<20} {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed! Your API is working perfectly!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
