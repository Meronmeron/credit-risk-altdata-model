"""
Unit Tests for Data Processing Module

This module contains unit tests for:
- Data loading functionality
- RFM calculation
- Risk proxy creation
- Feature engineering
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """
    Test cases for DataProcessor class
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.processor = DataProcessor()
        
        # Create sample transaction data
        self.sample_data = pd.DataFrame({
            'customer_id': ['C001', 'C001', 'C002', 'C002', 'C003'],
            'transaction_date': [
                '2024-01-01', '2024-01-15', '2024-01-05', 
                '2024-01-20', '2024-01-10'
            ],
            'amount': [100, 150, 200, 50, 300]
        })
        
        # Convert dates
        self.sample_data['transaction_date'] = pd.to_datetime(
            self.sample_data['transaction_date']
        )
    
    def test_calculate_rfm(self):
        """
        Test RFM calculation functionality
        """
        rfm_result = self.processor.calculate_rfm(
            self.sample_data, 
            'customer_id', 
            'transaction_date', 
            'amount'
        )
        
        # Check if RFM dataframe has correct columns
        expected_columns = ['Recency', 'Frequency', 'Monetary']
        self.assertTrue(all(col in rfm_result.columns for col in expected_columns))
        
        # Check if all customers are present
        self.assertEqual(len(rfm_result), 3)
        
        # Check data types
        self.assertTrue(rfm_result['Recency'].dtype in ['int64', 'float64'])
        self.assertTrue(rfm_result['Frequency'].dtype in ['int64', 'float64'])
        self.assertTrue(rfm_result['Monetary'].dtype in ['int64', 'float64'])
    
    def test_create_risk_proxy(self):
        """
        Test risk proxy creation
        """
        # First calculate RFM
        rfm_data = self.processor.calculate_rfm(
            self.sample_data, 
            'customer_id', 
            'transaction_date', 
            'amount'
        )
        
        # Create risk proxy
        risk_proxy_result = self.processor.create_risk_proxy(rfm_data)
        
        # Check if Risk_Proxy column exists
        self.assertIn('Risk_Proxy', risk_proxy_result.columns)
        
        # Check if values are binary (0 or 1)
        unique_values = risk_proxy_result['Risk_Proxy'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))
        
        # Check if RFM scores are created
        score_columns = ['R_Score', 'F_Score', 'M_Score', 'RFM_Score']
        self.assertTrue(all(col in risk_proxy_result.columns for col in score_columns))
    
    def test_data_loading_csv(self):
        """
        Test CSV data loading (mock test)
        """
        # Create a temporary CSV file for testing
        temp_file = 'temp_test.csv'
        self.sample_data.to_csv(temp_file, index=False)
        
        try:
            loaded_data = self.processor.load_data(temp_file)
            
            # Check if data is loaded correctly
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_invalid_file_format(self):
        """
        Test handling of invalid file formats
        """
        with self.assertRaises(ValueError):
            self.processor.load_data('invalid_file.txt')
    
    def test_rfm_with_empty_data(self):
        """
        Test RFM calculation with empty data
        """
        empty_data = pd.DataFrame(columns=['customer_id', 'transaction_date', 'amount'])
        
        with self.assertRaises(Exception):
            self.processor.calculate_rfm(empty_data, 'customer_id', 'transaction_date', 'amount')
    
    def test_rfm_monetary_calculation(self):
        """
        Test monetary value calculation in RFM
        """
        rfm_result = self.processor.calculate_rfm(
            self.sample_data, 
            'customer_id', 
            'transaction_date', 
            'amount'
        )
        
        # Check monetary values
        expected_monetary = {
            'C001': 250,  # 100 + 150
            'C002': 250,  # 200 + 50
            'C003': 300   # 300
        }
        
        for customer_id, expected_value in expected_monetary.items():
            actual_value = rfm_result.loc[customer_id, 'Monetary']
            self.assertEqual(actual_value, expected_value)
    
    def test_rfm_frequency_calculation(self):
        """
        Test frequency calculation in RFM
        """
        rfm_result = self.processor.calculate_rfm(
            self.sample_data, 
            'customer_id', 
            'transaction_date', 
            'amount'
        )
        
        # Check frequency values
        expected_frequency = {
            'C001': 2,  # 2 transactions
            'C002': 2,  # 2 transactions
            'C003': 1   # 1 transaction
        }
        
        for customer_id, expected_value in expected_frequency.items():
            actual_value = rfm_result.loc[customer_id, 'Frequency']
            self.assertEqual(actual_value, expected_value)


class TestDataProcessorIntegration(unittest.TestCase):
    """
    Integration tests for DataProcessor
    """
    
    def setUp(self):
        """
        Set up test fixtures for integration tests
        """
        self.processor = DataProcessor()
        
        # Create larger sample dataset
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        self.large_sample = pd.DataFrame({
            'customer_id': np.random.choice(['C001', 'C002', 'C003', 'C004', 'C005'], 1000),
            'transaction_date': np.random.choice(dates, 1000),
            'amount': np.random.uniform(10, 1000, 1000)
        })
    
    def test_end_to_end_pipeline(self):
        """
        Test the complete data processing pipeline
        """
        # Step 1: Calculate RFM
        rfm_data = self.processor.calculate_rfm(
            self.large_sample,
            'customer_id',
            'transaction_date', 
            'amount'
        )
        
        # Step 2: Create risk proxy
        risk_data = self.processor.create_risk_proxy(rfm_data)
        
        # Verify the complete pipeline
        self.assertIsNotNone(risk_data)
        self.assertIn('Risk_Proxy', risk_data.columns)
        
        # Check that we have reasonable distribution of risk categories
        risk_distribution = risk_data['Risk_Proxy'].value_counts()
        self.assertTrue(len(risk_distribution) > 0)


if __name__ == '__main__':
    unittest.main() 