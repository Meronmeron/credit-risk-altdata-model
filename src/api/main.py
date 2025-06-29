"""
FastAPI Application for Credit Risk Model

This module provides REST API endpoints for:
- Risk probability prediction
- Credit score calculation
- Loan term optimization
- Model health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import os
from datetime import datetime

from .pydantic_models import (
    CustomerFeatures, 
    RiskPredictionResponse,
    LoanRecommendationResponse,
    BatchPredictionRequest,
    HealthCheckResponse
)
from ..predict import CreditRiskPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="API for credit risk assessment and loan optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: CreditRiskPredictor = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    global predictor
    
    try:
        model_path = os.getenv("MODEL_PATH", "models/credit_risk_model.joblib")
        
        if os.path.exists(model_path):
            predictor = CreditRiskPredictor(model_path)
            logger.info("Model loaded successfully")
        else:
            predictor = CreditRiskPredictor()
            logger.warning("Model file not found")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        predictor = CreditRiskPredictor()


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Credit Risk Model API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    model_loaded = predictor is not None and predictor.model is not None
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(customer: CustomerFeatures):
    """
    Predict risk probability for a single customer
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer.dict()])
        
        # Get prediction
        risk_prob = predictor.predict_risk_probability(customer_df)[0]
        credit_score = predictor.calculate_credit_score(np.array([risk_prob]))[0]
        risk_level = predictor.categorize_risk_level(np.array([credit_score]))[0]
        
        return RiskPredictionResponse(
            customer_id=customer.customer_id,
            risk_probability=float(risk_prob),
            credit_score=int(credit_score),
            risk_level=risk_level,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/loan-terms", response_model=LoanRecommendationResponse)
async def predict_loan_terms(customer: CustomerFeatures):
    """
    Get loan term recommendations for a customer
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Get comprehensive recommendation
        recommendation = predictor.generate_loan_recommendation(customer.dict())
        
        return LoanRecommendationResponse(**recommendation)
        
    except Exception as e:
        logger.error(f"Loan recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint for multiple customers
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Convert customers to DataFrame
        customers_data = [customer.dict() for customer in request.customers]
        customers_df = pd.DataFrame(customers_data)
        
        # Get predictions
        risk_probs = predictor.predict_risk_probability(customers_df)
        credit_scores = predictor.calculate_credit_score(risk_probs)
        risk_levels = predictor.categorize_risk_level(credit_scores)
        
        # Prepare results
        results = []
        for i, customer in enumerate(request.customers):
            result = {
                "customer_id": customer.customer_id,
                "risk_probability": float(risk_probs[i]),
                "credit_score": int(credit_scores[i]),
                "risk_level": risk_levels[i]
            }
            results.append(result)
        
        return {
            "batch_id": request.batch_id,
            "processed_count": len(results),
            "results": results,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """
    Get model information and metadata
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        info = {
            "model_type": type(predictor.model).__name__,
            "feature_count": len(predictor.feature_names) if predictor.feature_names else 0,
            "feature_names": predictor.feature_names,
            "performance_metrics": getattr(predictor, 'performance_metrics', {}),
            "model_loaded": True
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 