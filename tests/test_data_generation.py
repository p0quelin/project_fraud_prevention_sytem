"""
Tests for the data generation functionality.
"""

import os
import sys
import pandas as pd
import pytest

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the generator functions
from generator import generate_customer_profiles_table, generate_terminal_profiles_table, generate_transactions

def test_customer_profiles_generation():
    """Test that customer profiles are generated with the right structure."""
    n_customers = 10
    customer_profiles = generate_customer_profiles_table(n_customers)
    
    # Assert the dataframe has the right number of rows
    assert len(customer_profiles) == n_customers
    
    # Assert the dataframe has the required columns
    required_columns = ['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount']
    for col in required_columns:
        assert col in customer_profiles.columns

def test_terminal_profiles_generation():
    """Test that terminal profiles are generated with the right structure."""
    n_terminals = 10
    terminal_profiles = generate_terminal_profiles_table(n_terminals)
    
    # Assert the dataframe has the right number of rows
    assert len(terminal_profiles) == n_terminals
    
    # Assert the dataframe has the required columns
    required_columns = ['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id', 'terminal_type']
    for col in required_columns:
        assert col in terminal_profiles.columns
    
    # Assert that terminal types are one of the expected values
    valid_types = ['atm', 'pos', 'online', 'retail']
    for term_type in terminal_profiles['terminal_type']:
        assert term_type in valid_types

def test_transactions_generation():
    """Test that transactions are generated properly."""
    # Generate small test data
    n_customers = 5
    n_terminals = 10
    nb_days = 5
    
    customer_profiles = generate_customer_profiles_table(n_customers)
    terminal_profiles = generate_terminal_profiles_table(n_terminals)
    
    # Generate transactions
    transactions = generate_transactions(customer_profiles, terminal_profiles, nb_days, 
                                        start_date="2020-01-01", r=5)
    
    # Assert transactions dataframe is not empty
    assert not transactions.empty
    
    # Assert the dataframe has the required columns
    required_columns = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 
                        'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_FRAUD']
    for col in required_columns:
        assert col in transactions.columns
    
    # Assert that some transactions are marked as fraud
    assert transactions['TX_FRAUD'].sum() > 0 