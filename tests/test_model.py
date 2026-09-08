"""
Tests for model prediction module
"""
import pytest
import numpy as np
import pandas as pd
from src.models.predict import FraudPredictor


@pytest.fixture
def predictor():
    """Load the trained model once for all tests"""
    return FraudPredictor()


def test_model_loads_successfully(predictor):
    """Test that the model loads without errors"""
    assert predictor.model is not None
    assert predictor.metadata is not None
    assert predictor.model_version is not None


def test_model_has_correct_threshold(predictor):
    """Test that the model uses the expected threshold"""
    assert predictor.threshold == 0.70


def test_prediction_returns_correct_structure(predictor):
    """Test that predict() returns the expected dictionary structure"""
    # Create a minimal transaction with all required features
    transaction = {feature: 0.0 for feature in predictor.feature_names}
    transaction['Amount'] = 100.0
    transaction['amount_log'] = np.log1p(100.0)

    result = predictor.predict(transaction)

    # Check that all expected keys are present
    assert 'fraud_probability' in result
    assert 'prediction' in result
    assert 'risk_level' in result
    assert 'model_version' in result


def test_prediction_probability_range(predictor):
    """Test that fraud probability is between 0 and 1"""
    transaction = {feature: 0.0 for feature in predictor.feature_names}
    transaction['Amount'] = 100.0
    transaction['amount_log'] = np.log1p(100.0)

    result = predictor.predict(transaction)

    assert 0.0 <= result['fraud_probability'] <= 1.0


def test_prediction_label_valid(predictor):
    """Test that prediction is either FRAUD or LEGITIMATE"""
    transaction = {feature: 0.0 for feature in predictor.feature_names}
    transaction['Amount'] = 100.0
    transaction['amount_log'] = np.log1p(100.0)

    result = predictor.predict(transaction)

    assert result['prediction'] in ['FRAUD', 'LEGITIMATE']


def test_risk_level_valid(predictor):
    """Test that risk_level is one of the expected values"""
    transaction = {feature: 0.0 for feature in predictor.feature_names}
    transaction['Amount'] = 100.0
    transaction['amount_log'] = np.log1p(100.0)

    result = predictor.predict(transaction)

    assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


def test_high_probability_triggers_fraud_prediction(predictor):
    """Test that high fraud probability results in FRAUD prediction"""
    # This test might not always pass depending on the model
    # but demonstrates the expected behavior
    transaction = {feature: 0.0 for feature in predictor.feature_names}
    transaction['Amount'] = 100.0
    transaction['amount_log'] = np.log1p(100.0)

    result = predictor.predict(transaction)

    # If probability is above threshold, should be FRAUD
    if result['fraud_probability'] >= predictor.threshold:
        assert result['prediction'] == 'FRAUD'
    else:
        assert result['prediction'] == 'LEGITIMATE'


def test_model_version_matches_metadata(predictor):
    """Test that model version is consistent"""
    assert predictor.model_version == predictor.metadata['model_version']
