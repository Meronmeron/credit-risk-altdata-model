"""
Model Training Module

This module contains functions for:
- Model training and validation
- Hyperparameter tuning
- Model evaluation and metrics
- Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Credit Risk Model Training and Evaluation
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.performance_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test split ratio
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_type: str = 'random_forest') -> None:
        """
        Train the credit risk model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_type (str): Type of model to train
        """
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        logger.info(f"{model_type} model trained successfully")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict: Performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        self.performance_metrics = {
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Model evaluation completed. AUC Score: {auc_score:.4f}")
        return self.performance_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model does not have feature importance attribute")
            return pd.DataFrame()
    
    def predict_risk_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk probability for new data
        
        Args:
            X (np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Risk probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
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
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and scaler
        
        Args:
            filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data.get('performance_metrics', {})
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """
    Main training pipeline
    """
    # TODO: Implement main training pipeline
    logger.info("Training pipeline started")
    
    # Load processed data
    # Train model
    # Evaluate model
    # Save model
    
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main() 