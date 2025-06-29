# Training Plots and Visualizations

This directory contains visualizations from the model training process.

## Sample Outputs (Included in Git)

- `sample_model_comparison.png` - Performance metrics comparison across models
- `sample_roc_auc_comparison.png` - ROC-AUC scores for all models  
- `sample_confusion_matrix.png` - Confusion matrix for best model

## Full Results (Local Only)

Full training generates many plots that are excluded from Git:
- Individual model confusion matrices
- ROC curves for each model
- Feature importance plots  
- Learning curves
- Hyperparameter tuning results

## View Full Results

To see all generated plots:
```bash
# Run training
python src/train_models.py

# Plots saved to plots/ directory
# View in MLflow UI for interactive exploration
mlflow ui
```

## Plot Types Generated

1. **Model Comparison**: Side-by-side metric comparison
2. **Confusion Matrices**: True/false positive analysis
3. **ROC Curves**: Threshold performance analysis
4. **Feature Importance**: Most predictive features
5. **Learning Curves**: Training vs validation performance

---

*Sample plots are representative of actual training results*
