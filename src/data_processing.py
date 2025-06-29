# -*- coding: utf-8 -*-
"""
Comprehensive Feature Engineering Module

This module contains advanced feature engineering pipeline including:
- Aggregate features (RFM + statistical features)
- Temporal feature extraction  
- Categorical encoding (One-Hot, Label, WoE)
- Missing value imputation
- Feature scaling and normalization
- Weight of Evidence (WoE) and Information Value (IV)
- Automated preprocessing pipeline using sklearn
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, 
    OneHotEncoder, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# WoE and IV libraries
try:
    from xverse import WOE
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom Transformers
class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates aggregate features for each customer including RFM metrics
    """
    
    def __init__(self, customer_col='CustomerId', date_col='TransactionStartTime', 
                 amount_col='Amount', value_col='Value'):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.reference_date = None
        
    def fit(self, X, y=None):
        """Calculate reference date for recency"""
        if self.date_col in X.columns:
            X_copy = X.copy()
            X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])
            self.reference_date = X_copy[self.date_col].max() + timedelta(days=1)
        return self
    
    def transform(self, X):
        """Transform transaction data to customer-level aggregate features"""
        X_copy = X.copy()
        
        # Convert date column to datetime
        if self.date_col in X_copy.columns:
            X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])
        
        # Calculate aggregate features per customer
        agg_dict = {
            # RFM Features
            self.date_col: [
                lambda x: (self.reference_date - x.max()).days,  # Recency
                'count'  # Frequency (transaction count)
            ],
            self.amount_col: [
                'sum',      # Total transaction amount (Monetary)
                'mean',     # Average transaction amount
                'std',      # Standard deviation of amounts
                'min',      # Minimum transaction amount
                'max',      # Maximum transaction amount
                'median'    # Median transaction amount
            ]
        }
        
        # Add value column if available
        if self.value_col in X_copy.columns:
            agg_dict[self.value_col] = [
                'sum',      # Total transaction value
                'mean',     # Average transaction value
                'std'       # Standard deviation of values
            ]
        
        agg_features = X_copy.groupby(self.customer_col).agg(agg_dict)
        
        # Flatten column names
        if self.value_col in X_copy.columns:
            agg_features.columns = [
                'Recency', 'Frequency',
                'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount', 'Median_Amount',
                'Total_Value', 'Avg_Value', 'Std_Value'
            ]
        else:
            agg_features.columns = [
                'Recency', 'Frequency',
                'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount', 'Median_Amount'
            ]
        
        # Handle missing std values (customers with single transactions)
        agg_features['Std_Amount'].fillna(0, inplace=True)
        if 'Std_Value' in agg_features.columns:
            agg_features['Std_Value'].fillna(0, inplace=True)
        
        # Additional derived features
        agg_features['Amount_Range'] = agg_features['Max_Amount'] - agg_features['Min_Amount']
        agg_features['Amount_CV'] = agg_features['Std_Amount'] / (agg_features['Avg_Amount'] + 1e-10)
        
        if 'Avg_Value' in agg_features.columns:
            agg_features['Value_CV'] = agg_features['Std_Value'] / (agg_features['Avg_Value'] + 1e-10)
        
        # Replace inf values with 0
        agg_features = agg_features.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Aggregate features created for {len(agg_features)} customers")
        return agg_features


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from transaction data
    """
    
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract temporal features from transaction datetime"""
        X_copy = X.copy()
        
        if self.date_col not in X_copy.columns:
            logger.warning(f"Date column {self.date_col} not found")
            return X_copy
        
        # Convert to datetime
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])
        
        # Extract temporal features
        X_copy['Transaction_Hour'] = X_copy[self.date_col].dt.hour
        X_copy['Transaction_Day'] = X_copy[self.date_col].dt.day
        X_copy['Transaction_Month'] = X_copy[self.date_col].dt.month
        X_copy['Transaction_Year'] = X_copy[self.date_col].dt.year
        X_copy['Transaction_DayOfWeek'] = X_copy[self.date_col].dt.dayofweek
        X_copy['Transaction_Quarter'] = X_copy[self.date_col].dt.quarter
        
        # Cyclical encoding for temporal features (better for ML models)
        X_copy['Hour_Sin'] = np.sin(2 * np.pi * X_copy['Transaction_Hour'] / 24)
        X_copy['Hour_Cos'] = np.cos(2 * np.pi * X_copy['Transaction_Hour'] / 24)
        X_copy['Month_Sin'] = np.sin(2 * np.pi * X_copy['Transaction_Month'] / 12)
        X_copy['Month_Cos'] = np.cos(2 * np.pi * X_copy['Transaction_Month'] / 12)
        X_copy['DayOfWeek_Sin'] = np.sin(2 * np.pi * X_copy['Transaction_DayOfWeek'] / 7)
        X_copy['DayOfWeek_Cos'] = np.cos(2 * np.pi * X_copy['Transaction_DayOfWeek'] / 7)
        
        logger.info("Temporal features extracted")
        return X_copy


class RiskProxyTransformer(BaseEstimator, TransformerMixin):
    """
    Creates risk proxy variable based on RFM analysis
    """
    
    def __init__(self, rfm_cols=['Recency', 'Frequency', 'Total_Amount']):
        self.rfm_cols = rfm_cols
        self.rfm_quantiles = {}
        
    def fit(self, X, y=None):
        """Calculate RFM quantiles for scoring"""
        for col in self.rfm_cols:
            if col in X.columns:
                if col == 'Recency':
                    # For recency, lower is better (reverse scoring)
                    self.rfm_quantiles[col] = X[col].quantile([0.2, 0.4, 0.6, 0.8])
                else:
                    # For frequency and monetary, higher is better
                    self.rfm_quantiles[col] = X[col].quantile([0.2, 0.4, 0.6, 0.8])
        return self
    
    def transform(self, X):
        """Create RFM scores and risk proxy"""
        X_copy = X.copy()
        
        # Create RFM scores (1-5 scale)
        for col in self.rfm_cols:
            if col in X_copy.columns:
                if col == 'Recency':
                    # For recency, lower values get higher scores
                    X_copy[f'{col}_Score'] = pd.cut(
                        X_copy[col], 
                        bins=[-np.inf] + list(self.rfm_quantiles[col]) + [np.inf],
                        labels=[5, 4, 3, 2, 1]
                    ).astype(int)
                else:
                    # For frequency and monetary, higher values get higher scores
                    X_copy[f'{col}_Score'] = pd.cut(
                        X_copy[col], 
                        bins=[-np.inf] + list(self.rfm_quantiles[col]) + [np.inf],
                        labels=[1, 2, 3, 4, 5]
                    ).astype(int)
        
        # Calculate overall RFM score
        score_cols = [f'{col}_Score' for col in self.rfm_cols if f'{col}_Score' in X_copy.columns]
        if score_cols:
            X_copy['RFM_Score'] = X_copy[score_cols].sum(axis=1)
        
        # Create Risk Proxy (0 = Good, 1 = Bad)
        # Customers in bottom 30% of RFM scores are considered high risk
        if 'RFM_Score' in X_copy.columns:
            risk_threshold = X_copy['RFM_Score'].quantile(0.3)
            X_copy['Risk_Proxy'] = (X_copy['RFM_Score'] <= risk_threshold).astype(int)
        
        logger.info("Risk proxy created based on RFM analysis")
        return X_copy


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) transformer for categorical variables
    """
    
    def __init__(self, categorical_cols: List[str], target_col: str = 'Risk_Proxy'):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.woe_mappings = {}
        self.iv_scores = {}
        
    def fit(self, X, y=None):
        """Calculate WoE mappings for categorical variables"""
        for col in self.categorical_cols:
            if col in X.columns and self.target_col in X.columns:
                self.woe_mappings[col], self.iv_scores[col] = self._calculate_woe_iv(
                    X[col], X[self.target_col]
                )
        return self
    
    def transform(self, X):
        """Apply WoE transformations"""
        X_copy = X.copy()
        
        for col in self.categorical_cols:
            if col in X_copy.columns and col in self.woe_mappings:
                woe_col = f'{col}_WoE'
                X_copy[woe_col] = X_copy[col].map(self.woe_mappings[col]).fillna(0)
        
        return X_copy
    
    def _calculate_woe_iv(self, feature: pd.Series, target: pd.Series) -> Tuple[Dict, float]:
        """Calculate Weight of Evidence and Information Value"""
        # Create crosstab
        crosstab = pd.crosstab(feature, target, margins=True)
        
        # Calculate WoE for each category
        woe_dict = {}
        iv = 0.0
        
        total_good = crosstab.loc['All', 0] if 0 in crosstab.columns else 1
        total_bad = crosstab.loc['All', 1] if 1 in crosstab.columns else 1
        
        for cat in crosstab.index[:-1]:  # Exclude 'All' row
            good = crosstab.loc[cat, 0] if 0 in crosstab.columns else 0.5
            bad = crosstab.loc[cat, 1] if 1 in crosstab.columns else 0.5
            
            # Add small constant to avoid division by zero
            good = max(good, 0.5)
            bad = max(bad, 0.5)
            
            dist_good = good / total_good
            dist_bad = bad / total_bad
            
            woe = np.log(dist_good / dist_bad)
            woe_dict[cat] = woe
            
            iv += (dist_good - dist_bad) * woe
        
        return woe_dict, iv


class KMeansRiskProxyTransformer(BaseEstimator, TransformerMixin):
    """
    Creates risk proxy variable using K-Means clustering on RFM features (Task 4)
    """
    
    def __init__(self, rfm_cols=['Recency', 'Frequency', 'Total_Amount'], 
                 n_clusters=3, random_state=42):
        self.rfm_cols = rfm_cols
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.high_risk_cluster = None
        self.cluster_profiles = None
        self._use_composite_score = False
        self._risk_threshold = None
        self._rfm_features_fit = None
        
    def fit(self, X, y=None):
        """Fit K-Means clustering on RFM features"""
        # Extract RFM features
        rfm_features = X[self.rfm_cols].copy()
        
        # Handle extreme outliers by capping values
        for col in self.rfm_cols:
            Q1 = rfm_features[col].quantile(0.25)
            Q3 = rfm_features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            rfm_features[col] = rfm_features[col].clip(lower_bound, upper_bound)
        
        # Scale features before clustering
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Fit K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(rfm_scaled)
        
        # Analyze clusters to identify high-risk segment
        cluster_analysis = rfm_features.copy()
        cluster_analysis['Cluster'] = cluster_labels
        
        # Calculate cluster profiles
        self.cluster_profiles = cluster_analysis.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Total_Amount': 'mean'
        })
        
        # Improved risk scoring logic
        # High-risk: High recency (inactive) + Low frequency + Low monetary
        risk_scores = []
        
        # Normalize each metric to 0-1 scale for fair comparison
        recency_min, recency_max = self.cluster_profiles['Recency'].min(), self.cluster_profiles['Recency'].max()
        frequency_min, frequency_max = self.cluster_profiles['Frequency'].min(), self.cluster_profiles['Frequency'].max()
        monetary_min, monetary_max = self.cluster_profiles['Total_Amount'].min(), self.cluster_profiles['Total_Amount'].max()
        
        for cluster in range(self.n_clusters):
            profile = self.cluster_profiles.loc[cluster]
            
            # Normalize scores (0-1 scale)
            if recency_max > recency_min:
                recency_norm = (profile['Recency'] - recency_min) / (recency_max - recency_min)
            else:
                recency_norm = 0
                
            if frequency_max > frequency_min:
                frequency_norm = (profile['Frequency'] - frequency_min) / (frequency_max - frequency_min)
            else:
                frequency_norm = 0
                
            if monetary_max > monetary_min:
                monetary_norm = (profile['Total_Amount'] - monetary_min) / (monetary_max - monetary_min)
            else:
                monetary_norm = 0
            
            # Risk score: Higher recency = worse, Lower frequency = worse, Lower monetary = worse
            risk_score = (recency_norm * 0.4) + ((1 - frequency_norm) * 0.3) + ((1 - monetary_norm) * 0.3)
            risk_scores.append(risk_score)
        
        # Cluster with highest risk score is the high-risk cluster
        self.high_risk_cluster = np.argmax(risk_scores)
        
        # If risk distribution is too extreme (>95% or <5%), use median split approach
        temp_labels = (cluster_labels == self.high_risk_cluster).astype(int)
        risk_pct = temp_labels.mean()
        
        if risk_pct < 0.05 or risk_pct > 0.95:
            # Use RFM composite score approach as fallback
            rfm_composite = (
                (rfm_features['Recency'] / rfm_features['Recency'].max()) * 0.4 +
                (1 - rfm_features['Frequency'] / rfm_features['Frequency'].max()) * 0.3 +
                (1 - rfm_features['Total_Amount'] / rfm_features['Total_Amount'].max()) * 0.3
            )
            # Use top 30% as high risk
            risk_threshold = rfm_composite.quantile(0.7)
            self._use_composite_score = True
            self._risk_threshold = risk_threshold
            self._rfm_features_fit = rfm_features
            
            logger.info(f"Using composite RFM score approach (cluster distribution too extreme)")
        else:
            self._use_composite_score = False
        
        logger.info(f"K-Means clustering completed with {self.n_clusters} clusters")
        logger.info(f"High-risk cluster identified: Cluster {self.high_risk_cluster}")
        logger.info(f"Risk scores: {risk_scores}")
        logger.info(f"Cluster profiles:\n{self.cluster_profiles}")
        
        return self
    
    def transform(self, X):
        """Apply clustering and create is_high_risk target variable"""
        X_copy = X.copy()
        
        # Extract RFM features
        rfm_features = X_copy[self.rfm_cols].copy()
        
        # Apply same outlier handling as in fit
        for col in self.rfm_cols:
            if col in rfm_features.columns:
                Q1 = rfm_features[col].quantile(0.25)
                Q3 = rfm_features[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                rfm_features[col] = rfm_features[col].clip(lower_bound, upper_bound)
        
        if hasattr(self, '_use_composite_score') and self._use_composite_score:
            # Use composite RFM score approach
            rfm_composite = (
                (rfm_features['Recency'] / rfm_features['Recency'].max()) * 0.4 +
                (1 - rfm_features['Frequency'] / rfm_features['Frequency'].max()) * 0.3 +
                (1 - rfm_features['Total_Amount'] / rfm_features['Total_Amount'].max()) * 0.3
            )
            X_copy['is_high_risk'] = (rfm_composite >= self._risk_threshold).astype(int)
            X_copy['Cluster'] = -1  # Indicate composite scoring was used
            
            logger.info(f"Using composite RFM scoring approach")
        else:
            # Use K-Means clustering approach
            rfm_scaled = self.scaler.transform(rfm_features)
            cluster_labels = self.kmeans.predict(rfm_scaled)
            X_copy['Cluster'] = cluster_labels
            X_copy['is_high_risk'] = (cluster_labels == self.high_risk_cluster).astype(int)
        
        risk_count = X_copy['is_high_risk'].sum()
        risk_pct = risk_count / len(X_copy) * 100
        
        logger.info(f"Risk proxy created using K-Means clustering")
        logger.info(f"High-risk customers: {risk_count} / {len(X_copy)} ({risk_pct:.1f}%)")
        
        return X_copy
    
    def get_cluster_analysis(self) -> pd.DataFrame:
        """Get detailed cluster analysis"""
        if self.cluster_profiles is None:
            return pd.DataFrame()
        
        analysis = self.cluster_profiles.copy()
        analysis['Is_High_Risk'] = 0
        analysis.loc[self.high_risk_cluster, 'Is_High_Risk'] = 1
        analysis['Risk_Level'] = ['Low', 'Medium', 'High'][::1] if self.n_clusters == 3 else [f'Cluster_{i}' for i in range(self.n_clusters)]
        
        # Add cluster size information if available
        return analysis


class ComprehensiveFeatureEngineering:
    """
    Comprehensive feature engineering pipeline using sklearn Pipeline
    """
    
    def __init__(self, customer_col='CustomerId', date_col='TransactionStartTime',
                 amount_col='Amount', value_col='Value'):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.customer_pipeline = None
        self.transaction_pipeline = None
        self.woe_transformer = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file"""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and date columns from categorical
        categorical_cols = [col for col in categorical_cols 
                          if col not in [self.customer_col, self.date_col]]
        
        # Remove ID and date columns from numerical
        numerical_cols = [col for col in numerical_cols 
                         if col not in [self.customer_col]]
        
        logger.info(f"Numerical columns: {numerical_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        return numerical_cols, categorical_cols
    
    def build_customer_pipeline(self, include_woe: bool = True) -> Pipeline:
        """Build sklearn pipeline for customer-level feature engineering"""
        steps = [
            ('aggregate', AggregateFeatureTransformer(
                self.customer_col, self.date_col, self.amount_col, self.value_col
            )),
            ('risk_proxy', RiskProxyTransformer()),
            ('scaler', StandardScaler())
        ]
        
        if include_woe:
            # WoE will be applied after we have categorical features
            pass
        
        self.customer_pipeline = Pipeline(steps)
        return self.customer_pipeline
    
    def fit_transform_customers(self, df: pd.DataFrame, include_woe: bool = True) -> pd.DataFrame:
        """Apply complete feature engineering pipeline to create customer-level features"""
        
        # Build and fit the pipeline
        pipeline = self.build_customer_pipeline(include_woe)
        
        # Apply transformations
        logger.info("Applying customer-level feature engineering...")
        
        # Step 1: Create aggregate features
        agg_transformer = AggregateFeatureTransformer(
            self.customer_col, self.date_col, self.amount_col, self.value_col
        )
        customer_features = agg_transformer.fit_transform(df)
        
        # Step 2: Create risk proxy
        risk_transformer = RiskProxyTransformer()
        customer_features = risk_transformer.fit_transform(customer_features)
        
        # Step 3: Apply WoE if requested (for any categorical features we might add)
        if include_woe:
            # Identify any categorical columns that might exist
            _, categorical_cols = self.identify_column_types(customer_features)
            if categorical_cols:
                woe_transformer = WoETransformer(categorical_cols)
                customer_features = woe_transformer.fit_transform(customer_features)
                self.woe_transformer = woe_transformer
        
        # Step 4: Scale numerical features (excluding target variable)
        numerical_cols = customer_features.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in ['Risk_Proxy']]
        
        if cols_to_scale:
            scaler = StandardScaler()
            customer_features[cols_to_scale] = scaler.fit_transform(customer_features[cols_to_scale])
        
        logger.info(f"Customer-level feature engineering completed: {customer_features.shape}")
        return customer_features
    
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-level features (for advanced modeling)"""
        logger.info("Creating transaction-level features...")
        
        # Apply temporal feature extraction
        temporal_transformer = TemporalFeatureTransformer(self.date_col)
        df_with_temporal = temporal_transformer.fit_transform(df)
        
        # Identify numerical and categorical columns
        numerical_cols, categorical_cols = self.identify_column_types(df_with_temporal)
        
        # Build preprocessing pipeline for transaction-level data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
                ]), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Apply preprocessing
        processed_features = preprocessor.fit_transform(df_with_temporal)
        
        # Create feature names
        feature_names = numerical_cols.copy()
        if categorical_cols:
            cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Add remaining column names
        remaining_cols = [col for col in df_with_temporal.columns 
                         if col not in numerical_cols + categorical_cols]
        feature_names.extend(remaining_cols)
        
        result_df = pd.DataFrame(processed_features, columns=feature_names)
        
        logger.info(f"Transaction-level features created: {result_df.shape}")
        return result_df
    
    def get_feature_importance_by_iv(self) -> pd.DataFrame:
        """Get feature importance based on Information Value"""
        if self.woe_transformer is None:
            logger.warning("WoE transformer not fitted. Run fit_transform_customers first.")
            return pd.DataFrame()
        
        iv_scores = []
        for col, iv in self.woe_transformer.iv_scores.items():
            iv_scores.append({'Feature': col, 'Information_Value': iv})
        
        iv_df = pd.DataFrame(iv_scores).sort_values('Information_Value', ascending=False)
        return iv_df

    def fit_transform_customers_task4(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Task 4 specific feature engineering with K-Means clustering risk proxy"""
        
        logger.info("Applying Task 4: K-Means clustering for risk proxy creation...")
        
        # Step 1: Create aggregate features (RFM)
        agg_transformer = AggregateFeatureTransformer(
            self.customer_col, self.date_col, self.amount_col, self.value_col
        )
        customer_features = agg_transformer.fit_transform(df)
        
        # Step 2: Apply K-Means clustering for risk proxy (Task 4)
        kmeans_risk_transformer = KMeansRiskProxyTransformer(
            rfm_cols=['Recency', 'Frequency', 'Total_Amount'],
            n_clusters=3,
            random_state=42
        )
        customer_features = kmeans_risk_transformer.fit_transform(customer_features)
        
        # Store the transformer for analysis
        self.kmeans_transformer = kmeans_risk_transformer
        
        # Step 3: Scale numerical features (excluding target variables)
        numerical_cols = customer_features.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in ['is_high_risk', 'Cluster']]
        
        if cols_to_scale:
            scaler = StandardScaler()
            customer_features[cols_to_scale] = scaler.fit_transform(customer_features[cols_to_scale])
        
        logger.info(f"Task 4 feature engineering completed: {customer_features.shape}")
        return customer_features
    
    def get_cluster_analysis_task4(self) -> pd.DataFrame:
        """Get detailed cluster analysis for Task 4"""
        if hasattr(self, 'kmeans_transformer'):
            return self.kmeans_transformer.get_cluster_analysis()
        else:
            logger.warning("K-Means transformer not fitted. Run fit_transform_customers_task4 first.")
            return pd.DataFrame()


# Legacy support - maintain backward compatibility
class DataProcessor(ComprehensiveFeatureEngineering):
    """Legacy DataProcessor class for backward compatibility"""
    
    def __init__(self):
        super().__init__()
    
    def calculate_rfm(self, df: pd.DataFrame, customer_id: str, 
                      transaction_date: str, amount: str) -> pd.DataFrame:
        """Legacy RFM calculation method"""
        self.customer_col = customer_id
        self.date_col = transaction_date
        self.amount_col = amount
        
        agg_transformer = AggregateFeatureTransformer(customer_id, transaction_date, amount)
        rfm_features = agg_transformer.fit_transform(df)
        
        return rfm_features
    
    def create_risk_proxy(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Legacy risk proxy creation method"""
        risk_transformer = RiskProxyTransformer()
        return risk_transformer.fit_transform(rfm_df)


if __name__ == "__main__":
    # Demo/test the feature engineering pipeline
    print("✅ Comprehensive Feature Engineering Module loaded successfully!")
    print("📊 Available classes:")
    print("  - ComprehensiveFeatureEngineering: Main feature engineering pipeline")
    print("  - AggregateFeatureTransformer: Customer-level RFM and statistical features")
    print("  - TemporalFeatureTransformer: Time-based feature extraction")
    print("  - RiskProxyTransformer: Risk scoring and proxy target creation (Task 3)")
    print("  - KMeansRiskProxyTransformer: K-Means clustering risk proxy (Task 4)")
    print("  - WoETransformer: Weight of Evidence encoding")
    print("  - DataProcessor: Legacy compatibility class")
    
    # Task 4 Demonstration
    print("\n" + "="*60)
    print("TASK 4 DEMONSTRATION: K-MEANS CLUSTERING RISK PROXY")
    print("="*60)
    
    try:
        # Initialize feature engineering pipeline
        fe = ComprehensiveFeatureEngineering()
        
        # Load sample data if available
        import os
        if os.path.exists('data/raw/data.csv'):
            print("Loading data for Task 4 demonstration...")
            data = fe.load_data('data/raw/data.csv')
            print(f"Data loaded: {data.shape}")
            
            # Apply Task 4 feature engineering
            print("\nApplying Task 4 K-Means clustering approach...")
            customer_features_task4 = fe.fit_transform_customers_task4(data)
            
            print(f"\n📊 Task 4 Results:")
            print(f"Customer features shape: {customer_features_task4.shape}")
            print(f"Target variable: 'is_high_risk' (0=Low risk, 1=High risk)")
            
            # Show cluster analysis
            cluster_analysis = fe.get_cluster_analysis_task4()
            print(f"\n🔍 Cluster Analysis:")
            print(cluster_analysis)
            
            # Show risk distribution
            risk_dist = customer_features_task4['is_high_risk'].value_counts()
            total = len(customer_features_task4)
            print(f"\n📈 Risk Distribution:")
            print(f"Low Risk (0): {risk_dist.get(0, 0):,} ({risk_dist.get(0, 0)/total*100:.1f}%)")
            print(f"High Risk (1): {risk_dist.get(1, 0):,} ({risk_dist.get(1, 0)/total*100:.1f}%)")
            
            print(f"\n✅ Task 4 completed successfully!")
            print(f"📁 Features ready for model training with 'is_high_risk' target")
            
        else:
            print("No data file found. Task 4 demonstration requires data/raw/data.csv")
            
    except Exception as e:
        print(f"❌ Task 4 demonstration error: {e}")
    
    logger.info("Feature engineering module ready for use!") 