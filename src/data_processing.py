"""
Data Processing Module

This module contains functions for:
- Data cleaning and preprocessing
- Feature engineering 
- RFM analysis
- Credit risk proxy variable creation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main class for data processing and feature engineering
    """
    
    def __init__(self):
        self.rfm_scores = None
        self.risk_proxy = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_rfm(self, df: pd.DataFrame, 
                      customer_id: str, 
                      transaction_date: str, 
                      amount: str) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics
        
        Args:
            df (pd.DataFrame): Transaction data
            customer_id (str): Customer ID column name
            transaction_date (str): Transaction date column name  
            amount (str): Transaction amount column name
            
        Returns:
            pd.DataFrame: RFM metrics per customer
        """
        # Convert date column to datetime
        df[transaction_date] = pd.to_datetime(df[transaction_date])
        
        # Calculate reference date (latest date + 1 day)
        reference_date = df[transaction_date].max() + timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id).agg({
            transaction_date: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id: 'count',  # Frequency
            amount: 'sum'  # Monetary
        }).rename(columns={
            transaction_date: 'Recency',
            customer_id: 'Frequency', 
            amount: 'Monetary'
        })
        
        self.rfm_scores = rfm
        logger.info("RFM calculation completed")
        return rfm
    
    def create_risk_proxy(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk proxy variable based on RFM analysis
        
        Args:
            rfm_df (pd.DataFrame): RFM metrics dataframe
            
        Returns:
            pd.DataFrame: DataFrame with risk proxy variable
        """
        # Create RFM scores (1-5 scale)
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5,4,3,2,1])
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Create combined RFM score
        rfm_df['RFM_Score'] = (rfm_df['R_Score'].astype(str) + 
                              rfm_df['F_Score'].astype(str) + 
                              rfm_df['M_Score'].astype(str))
        
        # Define risk proxy based on RFM segments
        # High value: High F&M, Low R = Good (0)
        # Low value: Low F&M, High R = Bad (1)
        
        conditions = [
            (rfm_df['R_Score'] >= 4) & (rfm_df['F_Score'] >= 4) & (rfm_df['M_Score'] >= 4),  # Good
            (rfm_df['R_Score'] <= 2) & (rfm_df['F_Score'] <= 2) & (rfm_df['M_Score'] <= 2)   # Bad
        ]
        
        choices = [0, 1]  # 0 = Good, 1 = Bad
        
        rfm_df['Risk_Proxy'] = np.select(conditions, choices, default=0.5)  # 0.5 = Neutral
        
        # Convert neutral to binary based on RFM score threshold
        neutral_mask = rfm_df['Risk_Proxy'] == 0.5
        threshold = rfm_df.loc[neutral_mask, ['R_Score', 'F_Score', 'M_Score']].sum(axis=1).median()
        
        rfm_df.loc[neutral_mask, 'Risk_Proxy'] = np.where(
            rfm_df.loc[neutral_mask, ['R_Score', 'F_Score', 'M_Score']].sum(axis=1) >= threshold, 0, 1
        )
        
        self.risk_proxy = rfm_df
        logger.info("Risk proxy variable created")
        return rfm_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for model training
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        # TODO: Implement feature engineering logic
        logger.info("Feature engineering completed")
        return df 