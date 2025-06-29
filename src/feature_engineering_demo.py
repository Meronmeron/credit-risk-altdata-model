"""
Feature Engineering Demonstration Script

This script demonstrates how to use the comprehensive feature engineering pipeline
for credit risk modeling with the Xente dataset.
"""

import pandas as pd
import numpy as np
import logging
from data_processing import ComprehensiveFeatureEngineering
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main demonstration of feature engineering pipeline
    """
    print("="*60)
    print("COMPREHENSIVE FEATURE ENGINEERING DEMONSTRATION")
    print("="*60)
    
    # Initialize the feature engineering pipeline
    fe = ComprehensiveFeatureEngineering(
        customer_col='CustomerId',
        date_col='TransactionStartTime', 
        amount_col='Amount',
        value_col='Value'
    )
    
    try:
        # Step 1: Load data
        print("\n1. Loading Data...")
        data = fe.load_data('data/raw/data.csv')
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        # Step 2: Identify column types
        print("\n2. Identifying Column Types...")
        categorical_cols, numerical_cols = fe.identify_column_types(data)
        print(f"   Categorical columns: {categorical_cols}")
        print(f"   Numerical columns: {numerical_cols}")
        
        # Step 3: Create customer-level features
        print("\n3. Creating Customer-Level Features...")
        print("   This includes:")
        print("   - RFM analysis (Recency, Frequency, Monetary)")
        print("   - Statistical aggregates (mean, std, min, max, etc.)")
        print("   - Risk proxy variable creation")
        print("   - Weight of Evidence transformations")
        
        customer_features = fe.fit_transform_customers(data, include_woe=True)
        
        print(f"   Customer features shape: {customer_features.shape}")
        print(f"   Customer features columns: {list(customer_features.columns)}")
        
        # Step 4: Analyze feature importance using Information Value
        print("\n4. Analyzing Feature Importance (Information Value)...")
        iv_scores = fe.get_feature_importance_by_iv()
        
        if not iv_scores.empty:
            print("   Top features by Information Value:")
            print(iv_scores.head(10).to_string(index=False))
        else:
            print("   No WoE features available for IV analysis")
        
        # Step 5: Display sample of engineered features
        print("\n5. Sample of Engineered Features:")
        print(customer_features.head().to_string())
        
        # Step 6: Feature statistics
        print("\n6. Feature Engineering Summary:")
        print(f"   Total customers: {len(customer_features):,}")
        print(f"   Total features created: {customer_features.shape[1]}")
        print(f"   Risk proxy distribution:")
        
        risk_dist = customer_features['Risk_Proxy'].value_counts()
        for value, count in risk_dist.items():
            risk_type = "Good" if value == 0 else "Bad"
            percentage = (count / len(customer_features)) * 100
            print(f"     {risk_type} customers: {count:,} ({percentage:.1f}%)")
        
        # Step 7: RFM Analysis Summary
        print("\n7. RFM Analysis Summary:")
        rfm_cols = ['Recency', 'Frequency', 'Total_Amount']
        for col in rfm_cols:
            if col in customer_features.columns:
                print(f"   {col}:")
                print(f"     Mean: {customer_features[col].mean():.2f}")
                print(f"     Median: {customer_features[col].median():.2f}")
                print(f"     Std: {customer_features[col].std():.2f}")
        
        # Step 8: Save processed data
        print("\n8. Saving Processed Data...")
        output_path = '../data/processed/customer_features.csv'
        customer_features.to_csv(output_path)
        print(f"   Customer features saved to: {output_path}")
        
        # Optional: Create transaction-level features demo
        print("\n9. Creating Transaction-Level Features (Sample)...")
        # Take a sample for demonstration
        sample_data = data.sample(n=min(10000, len(data)), random_state=42)
        transaction_features = fe.create_transaction_features(sample_data)
        print(f"   Transaction features shape: {transaction_features.shape}")
        print(f"   Sample features: {list(transaction_features.columns[:10])}")
        
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING DEMONSTRATION COMPLETE!")
        print("="*60)
        
        # Return the engineered features for further use
        return customer_features, transaction_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering demonstration: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return None, None


def analyze_features(customer_features: pd.DataFrame):
    """
    Additional analysis of engineered features
    """
    if customer_features is None:
        return
        
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Correlation analysis
    print("\n1. Feature Correlations with Risk Proxy:")
    correlations = customer_features.corr()['Risk_Proxy'].sort_values(key=abs, ascending=False)
    
    print("   Top positive correlations (bad customers):")
    positive_corr = correlations[correlations > 0].head(5)
    for feature, corr in positive_corr.items():
        if feature != 'Risk_Proxy':
            print(f"     {feature}: {corr:.4f}")
    
    print("   Top negative correlations (good customers):")
    negative_corr = correlations[correlations < 0].head(5)
    for feature, corr in negative_corr.items():
        print(f"     {feature}: {corr:.4f}")
    
    # RFM Score distribution
    print("\n2. RFM Score Distribution:")
    if 'RFM_Score' in customer_features.columns:
        rfm_stats = customer_features['RFM_Score'].describe()
        print(f"   Min: {rfm_stats['min']}")
        print(f"   25th percentile: {rfm_stats['25%']}")
        print(f"   Median: {rfm_stats['50%']}")
        print(f"   75th percentile: {rfm_stats['75%']}")
        print(f"   Max: {rfm_stats['max']}")
    
    # Missing values check
    print("\n3. Missing Values Check:")
    missing_values = customer_features.isnull().sum()
    if missing_values.sum() > 0:
        print("   Features with missing values:")
        for feature, count in missing_values[missing_values > 0].items():
            percentage = (count / len(customer_features)) * 100
            print(f"     {feature}: {count} ({percentage:.1f}%)")
    else:
        print("   ✅ No missing values found!")


if __name__ == "__main__":
    # Run the demonstration
    customer_features, transaction_features = main()
    
    # Run additional analysis
    if customer_features is not None:
        analyze_features(customer_features)
        
        print("\n" + "="*60)
        print("📊 FEATURE ENGINEERING READY FOR MODEL TRAINING!")
        print("="*60)
        print("\nNext steps:")
        print("1. Use customer_features.csv for model training")
        print("2. Target variable: 'Risk_Proxy' (0=Good, 1=Bad)")
        print("3. Features are already scaled and preprocessed")
        print("4. Consider feature selection based on correlations and IV scores")
        print("5. Ready for train.py model training pipeline!") 