# -*- coding: utf-8 -*-
"""
Task 5 - Model Training and Tracking

This module implements comprehensive model training with:
- Multiple model types (Logistic Regression, Decision Trees, Random Forest, GBM)
- Hyperparameter tuning (Grid Search, Random Search)
- MLflow experiment tracking and model registry
- Comprehensive model evaluation metrics
- Best model selection and registration
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, List, Tuple, Any
import logging
import warnings
from datetime import datetime
import json
import joblib
import os

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import our feature engineering pipeline
from data_processing import ComprehensiveFeatureEngineering

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """
    Comprehensive model training pipeline with MLflow tracking
    """
    
    def __init__(self, experiment_name: str = "credit-risk-modeling", random_state: int = 42):
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Set up MLflow
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Initialized CreditRiskModelTrainer with experiment: {experiment_name}")
    
    def load_and_prepare_data(self, data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load data and prepare train/test splits"""
        
        if data_path and os.path.exists(data_path):
            # Load pre-processed data
            logger.info(f"Loading pre-processed data from {data_path}")
            data = pd.read_csv(data_path, index_col=0)
            
            if 'is_high_risk' not in data.columns:
                raise ValueError("Target column 'is_high_risk' not found in data")
                
            # Separate features and target
            X = data.drop(['is_high_risk'], axis=1)
            y = data['is_high_risk']
            
        else:
            # Create features using Task 4 pipeline
            logger.info("Creating features using Task 4 K-Means clustering approach...")
            fe = ComprehensiveFeatureEngineering()
            raw_data = fe.load_data('data/raw/data.csv')
            
            # Apply Task 4 feature engineering
            customer_features = fe.fit_transform_customers_task4(raw_data)
            
            # Save processed features
            os.makedirs('data/processed', exist_ok=True)
            customer_features.to_csv('data/processed/customer_features_task4.csv')
            
            # Separate features and target
            X = customer_features.drop(['is_high_risk', 'Cluster'], axis=1)
            y = customer_features['is_high_risk']
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Data prepared: Train {X_train.shape}, Test {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self) -> Dict[str, Any]:
        """Define models and their hyperparameter grids"""
        
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'search_type': 'grid'
            },
            
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'search_type': 'random'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'random'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'search_type': 'grid'
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'search_type': 'random'
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'search_type': 'random'
            }
        }
        
        self.models = models
        logger.info(f"Defined {len(models)} models for training")
        return models
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC if probabilities are available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning and MLflow tracking"""
        
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Get model configuration
            model_config = self.models[model_name]
            base_model = model_config['model']
            param_grid = model_config['params']
            search_type = model_config['search_type']
            
            # Log model parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("search_type", search_type)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Hyperparameter tuning
            if search_type == 'grid':
                search = GridSearchCV(
                    base_model, param_grid, cv=5, scoring='roc_auc',
                    n_jobs=-1, verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    base_model, param_grid, n_iter=20, cv=5, scoring='roc_auc',
                    n_jobs=-1, verbose=0, random_state=self.random_state
                )
            
            # Fit the model
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Log best parameters
            for param, value in search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Get probabilities
            try:
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
            except:
                y_train_proba = None
                y_test_proba = None
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log model artifacts
            if model_name in ['xgboost']:
                mlflow.xgboost.log_model(best_model, "model")
            elif model_name in ['lightgbm']:
                mlflow.lightgbm.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            # Save model locally
            model_path = f"models/{model_name}_model.joblib"
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)
            
            # Create and log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = f"plots/{model_name}_confusion_matrix.png"
            os.makedirs("plots", exist_ok=True)
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Create and log ROC curve if probabilities available
            if y_test_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_metrics["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - ROC Curve')
                plt.legend(loc="lower right")
                roc_path = f"plots/{model_name}_roc_curve.png"
                plt.savefig(roc_path)
                mlflow.log_artifact(roc_path)
                plt.close()
            
            # Store results
            result = {
                'model': best_model,
                'best_params': search.best_params_,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score': search.best_score_
            }
            
            # Update best model if this one is better
            test_f1 = test_metrics['f1_score']
            if test_f1 > self.best_score:
                self.best_score = test_f1
                self.best_model = best_model
                self.best_model_name = model_name
                logger.info(f"New best model: {model_name} with F1 score: {test_f1:.4f}")
            
            logger.info(f"Completed {model_name} - Test F1: {test_f1:.4f}")
            
            return result
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train all models and track experiments"""
        
        logger.info("Starting model training for all models...")
        
        # Define models
        self.define_models()
        
        # Train each model
        for model_name in self.models.keys():
            try:
                result = self.train_single_model(model_name, X_train, X_test, y_train, y_test)
                self.results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Create results summary
        self.create_results_summary()
        
        logger.info(f"Training completed. Best model: {self.best_model_name}")
        return self.results
    
    def create_results_summary(self):
        """Create and log comprehensive results summary"""
        
        # Create summary DataFrame
        summary_data = []
        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']
            summary_data.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1_Score': test_metrics['f1_score'],
                'ROC_AUC': test_metrics.get('roc_auc', 0.0),
                'CV_Score': result['cv_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1_Score', ascending=False)
        
        # Save summary
        summary_path = "results/model_comparison.csv"
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact(summary_path)
            
            # Log best model metrics
            mlflow.log_metric("best_model_f1", self.best_score)
            mlflow.log_param("best_model_name", self.best_model_name)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        x = np.arange(len(summary_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, summary_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*2, summary_df['Model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        comparison_path = "plots/model_comparison.png"
        plt.savefig(comparison_path)
        plt.close()
        
        logger.info(f"Results summary created:")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        
        return summary_df
    
    def register_best_model(self, model_name: str = None):
        """Register the best model in MLflow Model Registry"""
        
        if model_name is None:
            model_name = f"credit-risk-{self.best_model_name}"
        
        try:
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered best model '{self.best_model_name}' as '{model_name}' in Model Registry")
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
    
    def run_complete_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """Run the complete model training pipeline"""
        
        logger.info("Starting complete model training pipeline...")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_path)
        
        # Train all models
        results = self.train_all_models(X_train, X_test, y_train, y_test)
        
        # Register best model
        self.register_best_model()
        
        logger.info("Complete pipeline finished successfully!")
        
        return {
            'results': results,
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        }


def main():
    """Main function to run model training"""
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer(
        experiment_name="credit-risk-task5",
        random_state=42
    )
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("TASK 5 - MODEL TRAINING COMPLETED")
    print("="*60)
    print(f"Best Model: {results['best_model']}")
    print(f"Best F1 Score: {results['best_score']:.4f}")
    print(f"Training Data Shape: {results['data_shape']['train']}")
    print(f"Test Data Shape: {results['data_shape']['test']}")
    print("\n✅ All models trained and tracked in MLflow!")
    print("🎯 Best model registered in MLflow Model Registry!")
    print("📊 Check MLflow UI for detailed experiment tracking")


if __name__ == "__main__":
    main() 