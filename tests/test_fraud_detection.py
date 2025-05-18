"""
Tests for the fraud detection analysis functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock data for testing
@pytest.fixture
def mock_transactions():
    """Create mock transaction data for testing."""
    # Create a small dataframe with some transactions
    data = {
        'TRANSACTION_ID': range(1, 21),
        'TX_DATETIME': pd.date_range(start='2020-01-01', periods=20, freq='H'),
        'CUSTOMER_ID': np.random.choice(range(1, 6), 20),
        'TERMINAL_ID': np.random.choice(range(1, 11), 20),
        'TX_AMOUNT': np.random.uniform(10, 1000, 20),
        'TX_TIME_SECONDS': range(0, 20 * 3600, 3600),
        'TX_TIME_DAYS': np.arange(0, 20/24, 1/24),
        'TX_FRAUD': np.random.choice([0, 1], 20, p=[0.8, 0.2]),  # 20% fraud rate
        'TX_FRAUD_SCENARIO': [np.random.randint(1, 5) if f == 1 else 0 for f in np.random.choice([0, 1], 20, p=[0.8, 0.2])]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_customer_profiles():
    """Create mock customer profiles for testing."""
    data = {
        'CUSTOMER_ID': range(1, 6),
        'x_customer_id': np.random.uniform(0, 100, 5),
        'y_customer_id': np.random.uniform(0, 100, 5),
        'mean_amount': np.random.uniform(50, 500, 5),
        'std_amount': np.random.uniform(10, 100, 5),
        'available_terminals': [[1, 2, 3], [4, 5], [6, 7, 8], [9], [10]]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_terminal_profiles():
    """Create mock terminal profiles for testing."""
    data = {
        'TERMINAL_ID': range(1, 11),
        'x_terminal_id': np.random.uniform(0, 100, 10),
        'y_terminal_id': np.random.uniform(0, 100, 10),
        'terminal_type': np.random.choice(['atm', 'pos', 'online', 'retail'], 10)
    }
    return pd.DataFrame(data)

def test_feature_engineering():
    """Test that feature engineering functions correctly."""
    try:
        from fraud_detection_analysis import engineer_features
        
        # Load mock data
        transactions = mock_transactions()
        customer_profiles = mock_customer_profiles()
        terminal_profiles = mock_terminal_profiles()
        
        # Run feature engineering
        features_df = engineer_features(transactions, customer_profiles, terminal_profiles)
        
        # Check that features were created
        assert features_df is not None
        assert features_df.shape[0] == transactions.shape[0]  # Same number of rows
        assert features_df.shape[1] > transactions.shape[1]   # More columns due to new features
        
        # Check for specific engineered features
        expected_features = ['hour', 'day_of_week', 'is_weekend', 'is_night', 
                            'amount_mean_ratio', 'time_since_last_tx_hours']
        
        for feature in expected_features:
            assert feature in features_df.columns, f"Expected feature {feature} not found"
            
    except ImportError:
        pytest.skip("Could not import engineer_features function") 