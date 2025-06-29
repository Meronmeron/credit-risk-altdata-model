"""
Unit Tests for Data Processing Module

This module contains unit tests for:
- Data loading functionality
- RFM calculation
- Risk proxy creation
- Feature engineering
- K-Means clustering 
- Model training helpers
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_processing import DataProcessor, ComprehensiveFeatureEngineering, KMeansRiskProxyTransformer
except ImportError:
    # Fallback import
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_processing", 
                                                  os.path.join(os.path.dirname(__file__), '..', 'src', 'data_processing.py'))
    data_processing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_processing_module)
    
    DataProcessor = data_processing_module.DataProcessor
    ComprehensiveFeatureEngineering = getattr(data_processing_module, 'ComprehensiveFeatureEngineering', None)
    KMeansRiskProxyTransformer = getattr(data_processing_module, 'KMeansRiskProxyTransformer', None)


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


# NEW: Task 4 and Task 5 Unit Tests
class TestKMeansRiskProxyTransformer(unittest.TestCase):
    """
    Unit tests for Task 4 K-Means clustering risk proxy
    """
    
    def setUp(self):
        """Set up test fixtures for K-Means transformer"""
        self.transformer = KMeansRiskProxyTransformer(
            rfm_cols=['Recency', 'Frequency', 'Total_Amount'],
            n_clusters=3,
            random_state=42
        )
        
        # Create sample RFM data
        self.sample_rfm_data = pd.DataFrame({
            'Recency': [10, 30, 5, 50, 15, 25],
            'Frequency': [20, 5, 30, 2, 25, 8],
            'Total_Amount': [1000, 200, 1500, 100, 1200, 300]
        })
    
    def test_kmeans_transformer_initialization(self):
        """Test K-Means transformer initialization"""
        self.assertEqual(self.transformer.n_clusters, 3)
        self.assertEqual(self.transformer.random_state, 42)
        self.assertEqual(self.transformer.rfm_cols, ['Recency', 'Frequency', 'Total_Amount'])
        self.assertIsNone(self.transformer.kmeans)
        self.assertIsNone(self.transformer.scaler)
    
    def test_kmeans_fit_method(self):
        """Test K-Means transformer fit method"""
        # Fit the transformer
        self.transformer.fit(self.sample_rfm_data)
        
        # Check if components are fitted
        self.assertIsNotNone(self.transformer.kmeans)
        self.assertIsNotNone(self.transformer.scaler)
        self.assertIsNotNone(self.transformer.cluster_profiles)
        self.assertIsNotNone(self.transformer.high_risk_cluster)
        
        # Check cluster profiles shape
        self.assertEqual(self.transformer.cluster_profiles.shape[0], 3)  # 3 clusters
        self.assertEqual(self.transformer.cluster_profiles.shape[1], 3)  # 3 RFM features
    
    def test_kmeans_transform_method(self):
        """Test K-Means transformer transform method"""
        # Fit and transform
        self.transformer.fit(self.sample_rfm_data)
        result = self.transformer.transform(self.sample_rfm_data)
        
        # Check if required columns are added
        self.assertIn('is_high_risk', result.columns)
        self.assertIn('Cluster', result.columns)
        
        # Check data types
        self.assertTrue(result['is_high_risk'].dtype in ['int64', 'int32'])
        self.assertTrue(all(val in [0, 1] for val in result['is_high_risk'].unique()))
        
        # Check cluster assignments
        self.assertTrue(result['Cluster'].min() >= 0)
        self.assertTrue(result['Cluster'].max() < 3)
    
    def test_kmeans_get_cluster_analysis(self):
        """Test cluster analysis method"""
        self.transformer.fit(self.sample_rfm_data)
        analysis = self.transformer.get_cluster_analysis()
        
        # Check analysis structure
        self.assertIsInstance(analysis, pd.DataFrame)
        self.assertIn('Is_High_Risk', analysis.columns)
        self.assertIn('Risk_Level', analysis.columns)
        
        # Check that exactly one cluster is marked as high risk
        high_risk_count = analysis['Is_High_Risk'].sum()
        self.assertEqual(high_risk_count, 1)


class TestModelTrainingHelpers(unittest.TestCase):
    """
    Unit tests for Task 5 model training helper functions
    """
    
    def setUp(self):
        """Set up test fixtures for model training tests"""
        # Import the model training module
        try:
            from train_models import CreditRiskModelTrainer
            self.trainer_class = CreditRiskModelTrainer
        except ImportError:
            self.trainer_class = None
    
    def test_calculate_metrics_helper_function(self):
        """Test the calculate_metrics helper function"""
        if self.trainer_class is None:
            self.skipTest("CreditRiskModelTrainer not available")
        
        trainer = self.trainer_class(experiment_name="test_exp")
        
        # Create sample predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.4, 0.2, 0.9, 0.6, 0.1, 0.7])
        
        # Calculate metrics
        metrics = trainer.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check if all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(0 <= metrics[metric] <= 1)
    
    def test_calculate_metrics_without_probabilities(self):
        """Test calculate_metrics function without probability predictions"""
        if self.trainer_class is None:
            self.skipTest("CreditRiskModelTrainer not available")
        
        trainer = self.trainer_class(experiment_name="test_exp")
        
        # Create sample predictions without probabilities
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # Calculate metrics
        metrics = trainer.calculate_metrics(y_true, y_pred)
        
        # Check that metrics are calculated (ROC-AUC should be missing or 0)
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in basic_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_data_preparation_helper(self):
        """Test data preparation helper function"""
        if self.trainer_class is None:
            self.skipTest("CreditRiskModelTrainer not available")
        
        # Create sample data with is_high_risk target
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.uniform(0, 1, 100),
            'is_high_risk': np.random.choice([0, 1], 100)
        })
        
        # Save to temporary file
        temp_file = 'temp_test_data.csv'
        sample_data.to_csv(temp_file)
        
        try:
            trainer = self.trainer_class(experiment_name="test_exp")
            X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(temp_file)
            
            # Check shapes
            self.assertEqual(len(X_train) + len(X_test), len(sample_data))
            self.assertEqual(len(y_train), len(X_train))
            self.assertEqual(len(y_test), len(X_test))
            
            # Check that features don't include target
            self.assertNotIn('is_high_risk', X_train.columns)
            self.assertNotIn('is_high_risk', X_test.columns)
            
            # Check target values
            self.assertTrue(all(val in [0, 1] for val in y_train.unique()))
            self.assertTrue(all(val in [0, 1] for val in y_test.unique()))
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestComprehensiveFeatureEngineering(unittest.TestCase):
    """
    Unit tests for Task 4 ComprehensiveFeatureEngineering class
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.fe = ComprehensiveFeatureEngineering()
        
        # Create sample transaction data
        self.sample_transactions = pd.DataFrame({
            'CustomerId': ['C001', 'C001', 'C002', 'C002', 'C003'] * 10,
            'TransactionStartTime': pd.date_range('2024-01-01', periods=50, freq='D'),
            'Amount': np.random.uniform(10, 1000, 50),
            'Value': np.random.uniform(5, 500, 50)
        })
    
    def test_task4_feature_engineering_pipeline(self):
        """Test the complete Task 4 feature engineering pipeline"""
        # Apply Task 4 pipeline
        result = self.fe.fit_transform_customers_task4(self.sample_transactions)
        
        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if required columns are present
        required_columns = ['is_high_risk', 'Cluster']
        for col in required_columns:
            self.assertIn(col, result.columns)
        
        # Check target variable
        self.assertTrue(all(val in [0, 1] for val in result['is_high_risk'].unique()))
        
        # Check that we have customer-level data (not transaction-level)
        unique_customers = self.sample_transactions['CustomerId'].nunique()
        self.assertEqual(len(result), unique_customers)
    
    def test_cluster_analysis_task4(self):
        """Test cluster analysis for Task 4"""
        # Apply Task 4 pipeline
        self.fe.fit_transform_customers_task4(self.sample_transactions)
        
        # Get cluster analysis
        analysis = self.fe.get_cluster_analysis_task4()
        
        # Check analysis structure
        self.assertIsInstance(analysis, pd.DataFrame)
        
        # Should have cluster profiles if clustering was successful
        if not analysis.empty:
            self.assertIn('Is_High_Risk', analysis.columns)
    
    def test_identify_column_types_helper(self):
        """Test the identify_column_types helper function"""
        sample_data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical1': ['A', 'B', 'C', 'D', 'E'],
            'categorical2': ['X', 'Y', 'Z', 'X', 'Y'],
            'CustomerId': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
        
        numerical_cols, categorical_cols = self.fe.identify_column_types(sample_data)
        
        # Check numerical columns (should exclude CustomerId)
        self.assertIn('numeric1', numerical_cols)
        self.assertIn('numeric2', numerical_cols)
        self.assertNotIn('CustomerId', numerical_cols)
        
        # Check categorical columns (should exclude CustomerId and date columns)
        self.assertIn('categorical1', categorical_cols)
        self.assertIn('categorical2', categorical_cols)
        self.assertNotIn('CustomerId', categorical_cols)
        self.assertNotIn('TransactionStartTime', categorical_cols)


class TestRFMCalculationHelper(unittest.TestCase):
    """
    Unit tests for RFM calculation helper functions
    """
    
    def test_rfm_recency_calculation(self):
        """Test recency calculation logic"""
        # Create test data with known dates
        test_data = pd.DataFrame({
            'CustomerId': ['C001', 'C001', 'C002'],
            'TransactionStartTime': ['2024-01-01', '2024-01-15', '2024-01-10'],
            'Amount': [100, 200, 150]
        })
        test_data['TransactionStartTime'] = pd.to_datetime(test_data['TransactionStartTime'])
        
        from data_processing import AggregateFeatureTransformer
        
        transformer = AggregateFeatureTransformer()
        transformer.fit(test_data)
        result = transformer.transform(test_data)
        
        # Check that recency is calculated (should be non-negative integers)
        self.assertTrue(all(result['Recency'] >= 0))
        self.assertTrue(result['Recency'].dtype in ['int64', 'float64'])
        
        # Customer with more recent transaction should have lower recency
        c001_recency = result.loc['C001', 'Recency']
        c002_recency = result.loc['C002', 'Recency']
        
        # C001's last transaction (2024-01-15) is more recent than C002's (2024-01-10)
        self.assertLessEqual(c001_recency, c002_recency)
    
    def test_rfm_frequency_calculation(self):
        """Test frequency calculation logic"""
        # Create test data with known transaction counts
        test_data = pd.DataFrame({
            'CustomerId': ['C001', 'C001', 'C001', 'C002', 'C002'],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5, freq='D'),
            'Amount': [100, 200, 150, 300, 250]
        })
        
        from data_processing import AggregateFeatureTransformer
        
        transformer = AggregateFeatureTransformer()
        result = transformer.fit_transform(test_data)
        
        # Check frequency calculations
        c001_frequency = result.loc['C001', 'Frequency']
        c002_frequency = result.loc['C002', 'Frequency']
        
        self.assertEqual(c001_frequency, 3)  # C001 has 3 transactions
        self.assertEqual(c002_frequency, 2)  # C002 has 2 transactions


# NEW: Task 5 Unit Tests (Required)
class TestTask5ModelHelpers(unittest.TestCase):
    """
    Unit tests for Task 5 helper functions - Required for Task 5
    """
    
    def setUp(self):
        """Set up test fixtures"""
        # Import Task 5 components
        try:
            from train_models import CreditRiskModelTrainer
            self.trainer_class = CreditRiskModelTrainer
        except ImportError:
            self.trainer_class = None
    
    def test_calculate_metrics_helper_function(self):
        """Test the calculate_metrics helper function - Task 5 Requirement #1"""
        if self.trainer_class is None:
            self.skipTest("CreditRiskModelTrainer not available")
        
        trainer = self.trainer_class(experiment_name="test_exp")
        
        # Create sample predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.4, 0.2, 0.9, 0.6, 0.1, 0.7])
        
        # Calculate metrics
        metrics = trainer.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check if all required Task 5 metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            # Metrics should be between 0 and 1
            self.assertTrue(0 <= metrics[metric] <= 1)
        
        # Test accuracy calculation specifically
        from sklearn.metrics import accuracy_score
        expected_accuracy = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=5)
    
    def test_data_preparation_helper_function(self):
        """Test data preparation helper function - Task 5 Requirement #2"""
        if self.trainer_class is None:
            self.skipTest("CreditRiskModelTrainer not available")
        
        # Create sample data with is_high_risk target
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.uniform(0, 1, 100),
            'is_high_risk': np.random.choice([0, 1], 100)
        })
        
        # Save to temporary file
        temp_file = 'temp_test_data.csv'
        sample_data.to_csv(temp_file, index=False)
        
        try:
            trainer = self.trainer_class(experiment_name="test_exp")
            X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(temp_file)
            
            # Check shapes and data integrity
            self.assertEqual(len(X_train) + len(X_test), len(sample_data))
            self.assertEqual(len(y_train), len(X_train))
            self.assertEqual(len(y_test), len(X_test))
            
            # Check that target column is properly separated
            self.assertNotIn('is_high_risk', X_train.columns)
            self.assertNotIn('is_high_risk', X_test.columns)
            
            # Check target values are binary
            self.assertTrue(all(val in [0, 1] for val in y_train.unique()))
            self.assertTrue(all(val in [0, 1] for val in y_test.unique()))
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestKMeansRiskProxyTransformer(unittest.TestCase):
    """
    Unit tests for K-Means Risk Proxy Transformer - Task 4/5 Integration
    """
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from data_processing import KMeansRiskProxyTransformer
            self.transformer_class = KMeansRiskProxyTransformer
        except ImportError:
            self.transformer_class = None
        
        # Create sample RFM data
        self.sample_rfm_data = pd.DataFrame({
            'Recency': [10, 30, 5, 50, 15, 25],
            'Frequency': [20, 5, 30, 2, 25, 8],
            'Total_Amount': [1000, 200, 1500, 100, 1200, 300]
        })
    
    def test_kmeans_transformer_fit_transform(self):
        """Test K-Means transformer fit and transform methods"""
        if self.transformer_class is None:
            self.skipTest("KMeansRiskProxyTransformer not available")
        
        transformer = self.transformer_class(n_clusters=3, random_state=42)
        
        # Fit and transform
        result = transformer.fit_transform(self.sample_rfm_data)
        
        # Check if required columns are added
        self.assertIn('is_high_risk', result.columns)
        self.assertIn('Cluster', result.columns)
        
        # Check data types and values
        self.assertTrue(result['is_high_risk'].dtype in ['int64', 'int32'])
        self.assertTrue(all(val in [0, 1] for val in result['is_high_risk'].unique()))
        
        # Check cluster assignments
        unique_clusters = result['Cluster'].unique()
        self.assertTrue(len(unique_clusters) <= 3)  # Should have at most 3 clusters
        self.assertTrue(all(cluster >= 0 for cluster in unique_clusters))
    
    def test_cluster_analysis_method(self):
        """Test cluster analysis helper method"""
        if self.transformer_class is None:
            self.skipTest("KMeansRiskProxyTransformer not available")
        
        transformer = self.transformer_class(n_clusters=3, random_state=42)
        transformer.fit(self.sample_rfm_data)
        
        analysis = transformer.get_cluster_analysis()
        
        # Check analysis structure
        self.assertIsInstance(analysis, pd.DataFrame)
        if not analysis.empty:
            self.assertIn('Is_High_Risk', analysis.columns)


if __name__ == '__main__':
    unittest.main() 