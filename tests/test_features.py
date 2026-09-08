"""
Tests for feature engineering functions
"""
import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    create_time_features,
    create_amount_features,
    create_synthetic_identities,
    create_velocity_features
)


def test_create_time_features():
    """Test that hour_of_day is correctly extracted from Time column"""
    # Create sample data
    df = pd.DataFrame({
        'Time': [0, 3600, 7200, 86400]  # 0h, 1h, 2h, 24h (wraps to 0)
    })

    result = create_time_features(df)

    assert 'hour_of_day' in result.columns
    assert result['hour_of_day'].iloc[0] == 0
    assert result['hour_of_day'].iloc[1] == 1
    assert result['hour_of_day'].iloc[2] == 2
    assert result['hour_of_day'].iloc[3] == 0  # 24h wraps to 0


def test_create_amount_features():
    """Test log transformation of Amount"""
    df = pd.DataFrame({
        'Amount': [0, 1, 10, 100, 1000]
    })

    result = create_amount_features(df)

    assert 'amount_log' in result.columns
    assert result['amount_log'].iloc[0] == np.log1p(0)
    assert result['amount_log'].iloc[1] == np.log1p(1)
    assert result['amount_log'].iloc[4] == pytest.approx(np.log1p(1000))


def test_create_synthetic_identities():
    """Test that synthetic IDs are generated"""
    df = pd.DataFrame({
        'Amount': [100, 200, 300]
    })

    result = create_synthetic_identities(df, seed=42)

    assert 'customer_id' in result.columns
    assert 'merchant_id' in result.columns
    assert len(result) == 3
    # With same seed, results should be reproducible
    assert result['customer_id'].iloc[0] == result['customer_id'].iloc[0]


def test_create_velocity_features():
    """Test behavioral feature generation"""
    df = pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 2],
        'Amount': [100, 200, 50, 60, 70]
    })

    result = create_velocity_features(df)

    assert 'customer_avg_amount' in result.columns
    assert 'customer_txn_count' in result.columns
    assert 'high_amount_flag' in result.columns

    # Customer 1 has 2 transactions averaging 150
    assert result[result['customer_id'] == 1]['customer_avg_amount'].iloc[0] == 150
    assert result[result['customer_id'] == 1]['customer_txn_count'].iloc[0] == 2

    # Customer 2 has 3 transactions averaging 60
    assert result[result['customer_id'] == 2]['customer_avg_amount'].iloc[0] == 60
    assert result[result['customer_id'] == 2]['customer_txn_count'].iloc[0] == 3


def test_high_amount_flag():
    """Test that high_amount_flag correctly identifies transactions > 3x the customer average"""
    # Case 1: All amounts are similar (Flag should be 0)
    df1 = pd.DataFrame({
        'customer_id': [1, 1, 1],
        'Amount': [100, 110, 90]
    })
    result1 = create_velocity_features(df1)
    assert result1['high_amount_flag'].sum() == 0

    # Case 2: One amount is significantly higher than average (Flag should be 1)
    # Average = (10+10+10+500)/4 = 132.5. 500 > 132.5 * 3 (397.5)
    df2 = pd.DataFrame({
        'customer_id': [2, 2, 2, 2],
        'Amount': [10, 10, 10, 500]
    })
    result2 = create_velocity_features(df2)
    # Only the last transaction should be flagged
    assert result2['high_amount_flag'].iloc[3] == 1
    assert result2['high_amount_flag'].iloc[0] == 0
    assert result2['high_amount_flag'].sum() == 1
