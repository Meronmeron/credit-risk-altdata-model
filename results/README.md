# Model Training Results

This directory contains **summaries** of model training results. Full experiment data is stored locally in MLflow.

## Files Included in Git

### Keep in Repository:

- `model_comparison.csv` - Summary of all model performance metrics
- `training_summary.md` - Overview of training process and best results
- `sample_plots/` - Representative plots showing model performance
- Configuration files and training scripts

### Excluded from Repository (Local Only):

- `mlruns/` - Full MLflow experiment tracking data (can be GBs)
- `models/*.joblib` - Trained model files (large binary files)
- `plots/` - All generated plots (except samples)
- Detailed experiment logs

## How to Demonstrate Your Training Work

### 1. Repository Evidence

The code and configuration in this repo shows:

- Complete training pipeline implementation
- Multiple model types with hyperparameter tuning
- Comprehensive evaluation metrics
- MLflow integration setup
- Unit tests for critical functions

### 2. Training Summary

Check `training_summary.md` for:

- Model performance comparison
- Best model selection rationale
- Training process overview
- Key insights and findings

### 3. Local MLflow Results

To view full training results locally:

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
# View all experiments, parameters, metrics, and artifacts
```

### 4. Reproduce Training

Anyone can reproduce the training by:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python src/train_models.py

# View results in MLflow UI
mlflow ui
```

## Best Practices Followed

1. **Version Control**: Only code, configs, and summaries in Git
2. **Experiment Tracking**: Full tracking in MLflow (local/cloud)
3. **Reproducibility**: Seed setting and documented process
4. **Documentation**: Clear instructions and summaries
5. **Testing**: Unit tests for critical components
