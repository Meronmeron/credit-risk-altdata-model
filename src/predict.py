"""
Prediction Module

This module contains functions for:
- Model inference and prediction
- Risk probability calculation
- Credit score assignment
- Loan amount and duration optimization
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Credit Risk Prediction and Loan Optimization
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model and preprocessing components
        
        Args:
            model_path (str): Path to saved model
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_risk_probability(self, customer_features: pd.DataFrame) -> np.ndarray:
        """
        Predict risk probability for customers
        
        Args:
            customer_features (pd.DataFrame): Customer features
            
        Returns:
            np.ndarray: Risk probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure features are in correct order
        customer_features = customer_features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(customer_features)
        
        # Predict probabilities
        risk_probs = self.model.predict_proba(features_scaled)[:, 1]
        
        logger.info(f"Risk probabilities calculated for {len(customer_features)} customers")
        return risk_probs
    
    def calculate_credit_score(self, risk_probabilities: np.ndarray,
                             min_score: int = 300, max_score: int = 850) -> np.ndarray:
        """
        Convert risk probabilities to credit scores
        
        Args:
            risk_probabilities (np.ndarray): Risk probabilities
            min_score (int): Minimum credit score
            max_score (int): Maximum credit score
            
        Returns:
            np.ndarray: Credit scores
        """
        # Invert probabilities (lower risk = higher score)
        inverted_probs = 1 - risk_probabilities
        
        # Scale to credit score range
        credit_scores = min_score + (max_score - min_score) * inverted_probs
        
        return credit_scores.astype(int)
    
    def categorize_risk_level(self, credit_scores: np.ndarray) -> np.ndarray:
        """
        Categorize customers into risk levels based on credit scores
        
        Args:
            credit_scores (np.ndarray): Credit scores
            
        Returns:
            np.ndarray: Risk level categories
        """
        conditions = [
            credit_scores >= 750,  # Excellent
            (credit_scores >= 700) & (credit_scores < 750),  # Good
            (credit_scores >= 650) & (credit_scores < 700),  # Fair
            (credit_scores >= 600) & (credit_scores < 650),  # Poor
            credit_scores < 600    # Very Poor
        ]
        
        choices = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
        
        return np.select(conditions, choices, default='Unknown')
    
    def optimize_loan_terms(self, customer_features: pd.DataFrame,
                           base_amount: float = 10000,
                           base_duration: int = 12) -> pd.DataFrame:
        """
        Optimize loan amount and duration based on risk assessment
        
        Args:
            customer_features (pd.DataFrame): Customer features
            base_amount (float): Base loan amount
            base_duration (int): Base loan duration in months
            
        Returns:
            pd.DataFrame: Optimized loan terms
        """
        # Calculate risk probabilities and credit scores
        risk_probs = self.predict_risk_probability(customer_features)
        credit_scores = self.calculate_credit_score(risk_probs)
        risk_levels = self.categorize_risk_level(credit_scores)
        
        # Initialize loan terms
        loan_terms = pd.DataFrame({
            'customer_id': range(len(customer_features)),
            'risk_probability': risk_probs,
            'credit_score': credit_scores,
            'risk_level': risk_levels
        })
        
        # Optimize loan amount based on credit score
        loan_terms['max_loan_amount'] = self._calculate_max_loan_amount(
            credit_scores, base_amount
        )
        
        # Optimize loan duration based on risk level
        loan_terms['recommended_duration'] = self._calculate_recommended_duration(
            risk_levels, base_duration
        )
        
        # Calculate interest rate based on risk
        loan_terms['interest_rate'] = self._calculate_interest_rate(risk_probs)
        
        logger.info("Loan terms optimized for all customers")
        return loan_terms
    
    def _calculate_max_loan_amount(self, credit_scores: np.ndarray,
                                  base_amount: float) -> np.ndarray:
        """
        Calculate maximum loan amount based on credit score
        
        Args:
            credit_scores (np.ndarray): Credit scores
            base_amount (float): Base loan amount
            
        Returns:
            np.ndarray: Maximum loan amounts
        """
        # Score-based multipliers
        multipliers = np.where(credit_scores >= 750, 2.0,
                      np.where(credit_scores >= 700, 1.5,
                      np.where(credit_scores >= 650, 1.2,
                      np.where(credit_scores >= 600, 1.0, 0.5))))
        
        return (base_amount * multipliers).astype(int)
    
    def _calculate_recommended_duration(self, risk_levels: np.ndarray,
                                       base_duration: int) -> np.ndarray:
        """
        Calculate recommended loan duration based on risk level
        
        Args:
            risk_levels (np.ndarray): Risk level categories
            base_duration (int): Base duration in months
            
        Returns:
            np.ndarray: Recommended durations
        """
        duration_map = {
            'Excellent': base_duration * 2,    # 24 months
            'Good': int(base_duration * 1.5),  # 18 months
            'Fair': base_duration,             # 12 months
            'Poor': int(base_duration * 0.75), # 9 months
            'Very Poor': int(base_duration * 0.5)  # 6 months
        }
        
        return np.array([duration_map.get(level, base_duration) for level in risk_levels])
    
    def _calculate_interest_rate(self, risk_probabilities: np.ndarray,
                                base_rate: float = 0.05) -> np.ndarray:
        """
        Calculate interest rate based on risk probability
        
        Args:
            risk_probabilities (np.ndarray): Risk probabilities
            base_rate (float): Base interest rate
            
        Returns:
            np.ndarray: Interest rates
        """
        # Higher risk = higher interest rate
        risk_premium = risk_probabilities * 0.15  # Up to 15% premium
        interest_rates = base_rate + risk_premium
        
        return np.round(interest_rates, 4)
    
    def generate_loan_recommendation(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive loan recommendation for a single customer
        
        Args:
            customer_data (Dict): Customer features as dictionary
            
        Returns:
            Dict: Loan recommendation
        """
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Get loan terms
        loan_terms = self.optimize_loan_terms(customer_df)
        
        # Create recommendation
        recommendation = {
            'customer_id': customer_data.get('customer_id', 'N/A'),
            'risk_assessment': {
                'risk_probability': float(loan_terms['risk_probability'].iloc[0]),
                'credit_score': int(loan_terms['credit_score'].iloc[0]),
                'risk_level': loan_terms['risk_level'].iloc[0]
            },
            'loan_terms': {
                'max_amount': int(loan_terms['max_loan_amount'].iloc[0]),
                'recommended_duration': int(loan_terms['recommended_duration'].iloc[0]),
                'interest_rate': float(loan_terms['interest_rate'].iloc[0])
            },
            'recommendation': self._get_loan_decision(loan_terms['risk_level'].iloc[0])
        }
        
        return recommendation
    
    def _get_loan_decision(self, risk_level: str) -> str:
        """
        Get loan approval decision based on risk level
        
        Args:
            risk_level (str): Risk level category
            
        Returns:
            str: Loan decision
        """
        decision_map = {
            'Excellent': 'APPROVED - Premium terms available',
            'Good': 'APPROVED - Standard terms',
            'Fair': 'APPROVED - Conservative terms',
            'Poor': 'CONDITIONAL - Requires additional verification',
            'Very Poor': 'DECLINED - High risk'
        }
        
        return decision_map.get(risk_level, 'UNDER REVIEW')


def main():
    """
    Main prediction pipeline for batch processing
    """
    # TODO: Implement batch prediction pipeline
    logger.info("Prediction pipeline started")
    
    # Load model
    # Load customer data
    # Generate predictions
    # Save results
    
    logger.info("Prediction pipeline completed")


if __name__ == "__main__":
    main() 