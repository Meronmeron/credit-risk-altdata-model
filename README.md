# Credit Risk Model

A comprehensive credit risk assessment system that leverages behavioral data to predict customer creditworthiness and optimize loan terms.

## 🎯 Project Overview

This project builds a credit scoring model for Bati Bank's buy-now-pay-later service in partnership with an eCommerce platform. The system transforms customer behavioral data into predictive risk signals using RFM (Recency, Frequency, Monetary) analysis.

### Key Features

- **Risk Assessment**: Predicts default probability for new customers
- **Credit Scoring**: Converts risk probabilities to traditional credit scores (300-850)
- **Loan Optimization**: Recommends optimal loan amounts and durations
- **REST API**: Production-ready FastAPI interface
- **Scalable Architecture**: Containerized with Docker support

## 💼 Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability

The Basel II Accord's emphasis on risk measurement directly influences our modeling approach by requiring **transparent, auditable, and well-documented models**. Financial institutions must demonstrate to regulators that their credit risk models are not "black boxes" but have clear business logic that can be explained and validated. This regulatory requirement drives our need for:

- **Model Documentation**: Comprehensive explanation of feature selection, model assumptions, and business rationale
- **Interpretability**: Clear understanding of how each feature contributes to risk assessment decisions
- **Auditability**: Ability to trace and justify every credit decision made by the model
- **Validation Framework**: Robust backtesting and performance monitoring to ensure model reliability over time

### Proxy Variable Necessity and Business Risks

Since we lack direct "default" labels from historical loan performance, creating a proxy variable becomes **essential but introduces significant business risks**:

**Why Proxy Variables Are Necessary:**

- Traditional credit scoring requires historical default data, which we don't have for eCommerce customers
- RFM behavioral patterns can indicate creditworthiness (high frequency + high monetary + low recency = reliable customer)
- Regulatory compliance requires some form of risk classification to justify lending decisions

**Key Business Risks:**

- **Proxy Validity Risk**: Our behavioral proxy may not accurately reflect actual default probability, leading to systematic mispricing of risk
- **Population Shift Risk**: eCommerce behavior patterns may not translate to credit behavior, especially during economic stress
- **Regulatory Risk**: Regulators may challenge the validity of behavioral proxies for credit decisions
- **Adverse Selection**: Good eCommerce customers may be poor credit risks, and vice versa
- **Model Drift**: Behavioral patterns may change over time, making our proxy increasingly unreliable

### Model Complexity Trade-offs in Regulated Finance

The choice between simple and complex models in regulated financial contexts involves critical trade-offs:

**Simple Models (Logistic Regression with WoE):**

- ✅ **Regulatory Friendly**: Easy to explain to regulators and auditors
- ✅ **Interpretable**: Clear coefficient interpretation and business logic
- ✅ **Stable**: Less prone to overfitting and more robust across different market conditions
- ✅ **Transparent**: Stakeholders can understand and trust the decision process
- ❌ **Lower Performance**: May miss complex patterns and interactions in the data
- ❌ **Limited Adaptability**: Cannot capture non-linear relationships

**Complex Models (Gradient Boosting):**

- ✅ **Higher Performance**: Better predictive accuracy and ability to capture complex patterns
- ✅ **Adaptive**: Can learn intricate feature interactions and non-linear relationships
- ✅ **Competitive Advantage**: Superior performance can translate to better risk-adjusted returns
- ❌ **Regulatory Challenge**: Difficult to explain to regulators ("black box" perception)
- ❌ **Overfitting Risk**: May perform well on historical data but fail on new populations
- ❌ **Operational Complexity**: Harder to maintain, validate, and troubleshoot in production

**Recommended Approach:**
Given the regulated nature of financial services and our reliance on proxy variables, we advocate for a **hybrid approach**: starting with interpretable models for regulatory approval and business understanding, then gradually introducing complexity with robust validation and explainability tools (like SHAP values) to maintain transparency while improving performance.

## 🏗️ Architecture

```
credit-risk-model/
├── .github/workflows/      # CI/CD pipeline
├── data/                   # Data storage (gitignored)
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── data_processing.py  # Feature engineering
│   ├── train.py           # Model training
│   ├── predict.py         # Inference engine
│   └── api/               # FastAPI application
├── tests/                 # Unit tests
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service deployment
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd credit-risk-model
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place your raw data files in `data/raw/`
2. Run the data processing pipeline:

   ```python
   from src.data_processing import DataProcessor

   processor = DataProcessor()
   # Process your data
   ```

### Model Training

```bash
python src/train.py
```

### API Usage

1. **Start the API server**

   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker only
docker build -t credit-risk-model .
docker run -p 8000:8000 credit-risk-model
```

## 📊 Model Pipeline

### 1. Data Processing

- **RFM Calculation**: Analyzes customer Recency, Frequency, and Monetary patterns
- **Risk Proxy Creation**: Defines good/bad customers based on behavioral segments
- **Feature Engineering**: Creates predictive features from raw transaction data

### 2. Model Training

- **Algorithm Selection**: Supports Random Forest, Gradient Boosting, Logistic Regression
- **Hyperparameter Tuning**: Automated optimization with cross-validation
- **Performance Evaluation**: Comprehensive metrics including AUC, precision, recall

### 3. Inference

- **Risk Probability**: 0-1 probability of default
- **Credit Score**: Traditional 300-850 scale scoring
- **Loan Terms**: Optimized amount, duration, and interest rates

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

### Single Prediction

```http
POST /predict/risk
Content-Type: application/json

{
  "customer_id": "CUST_001",
  "recency": 30.0,
  "frequency": 15,
  "monetary": 5000.0
}
```

### Loan Recommendation

```http
POST /predict/loan-terms
Content-Type: application/json

{
  "customer_id": "CUST_001",
  "recency": 30.0,
  "frequency": 15,
  "monetary": 5000.0
}
```

### Batch Processing

```http
POST /predict/batch
Content-Type: application/json

{
  "batch_id": "BATCH_001",
  "customers": [...]
}
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Model Performance

The model performance metrics will be displayed here after training:

- **AUC-ROC**: Target > 0.75
- **Precision**: Target > 0.70
- **Recall**: Target > 0.65
- **F1-Score**: Target > 0.67

## 🔧 Configuration

### Environment Variables

```bash
MODEL_PATH=/path/to/model.joblib  # Path to trained model
LOG_LEVEL=info                    # Logging level
API_HOST=0.0.0.0                 # API host
API_PORT=8000                    # API port
```

### Model Parameters

Modify training parameters in `src/train.py`:

- `test_size`: Train/test split ratio
- `random_state`: Reproducibility seed
- `model_type`: Algorithm selection

## 📋 Basel II Compliance

This model considers Basel II Capital Accord requirements:

- **Credit Risk Assessment**: Quantitative default probability estimation
- **Risk-Based Pricing**: Interest rates adjusted for risk levels
- **Capital Adequacy**: Risk-weighted loan amount calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support, please contact:

- **Email**: analytics@batibank.com
- **Documentation**: [Project Wiki](wiki-url)
- **Issues**: [GitHub Issues](issues-url)

## 🔄 Roadmap

- [ ] Advanced feature engineering
- [ ] Model interpretability (SHAP values)
- [ ] A/B testing framework
- [ ] Real-time streaming predictions
- [ ] Integration with external credit bureaus
