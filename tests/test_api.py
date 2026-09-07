"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test that /health returns 200 OK"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_version" in data


def test_model_info_endpoint():
    """Test that /model-info returns model metadata"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "threshold" in data
    assert "performance_metrics" in data
    assert "feature_count" in data


def test_predict_endpoint_with_valid_data():
    """Test /predict with a valid transaction"""
    import time

    # Create a complete valid transaction with unique ID
    timestamp = int(time.time() * 1000)
    transaction = {
        "transaction_id": f"TEST{timestamp}",  # Unique ID using timestamp
        "Amount": 150.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "hour_of_day": 14,
        "amount_log": 5.01,
        "customer_avg_amount": 120.0,
        "customer_txn_count": 15,
        "high_amount_flag": 0
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    data = response.json()
    assert "transaction_id" in data
    assert data["transaction_id"] == transaction["transaction_id"]
    assert "prediction" in data
    assert data["prediction"] in ["FRAUD", "LEGITIMATE"]
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_endpoint_with_invalid_data():
    """Test /predict with missing required fields"""
    # Amount is required and has no default, so omitting it should cause validation error
    invalid_transaction = {
        "transaction_id": "TEST002"
        # Missing Amount (required field)
    }

    response = client.post("/predict", json=invalid_transaction)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_with_negative_amount():
    """Test /predict rejects negative amounts"""
    transaction = {
        "transaction_id": "TEST003",
        "Amount": -50.0,  # Invalid negative amount
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 422  # Validation error


def test_transactions_endpoint():
    """Test /transactions endpoint returns recent predictions"""
    response = client.get("/transactions?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "transactions" in data
    assert isinstance(data["transactions"], list)


def test_root_endpoint():
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
