"""
Pydantic Models for Credit Risk API

This module contains Pydantic models for:
- Request validation
- Response serialization
- Data type enforcement
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class CustomerFeatures(BaseModel):
    """
    Customer features for credit risk assessment
    """

    customer_id: str = Field(..., description="Unique customer identifier")
    recency: float = Field(..., description="Days since last transaction")
    frequency: int = Field(..., description="Number of transactions")
    monetary: float = Field(..., description="Total transaction amount")

    # Additional features can be added here
    age: Optional[int] = Field(None, description="Customer age")
    income: Optional[float] = Field(None, description="Monthly income")
    employment_length: Optional[int] = Field(
        None, description="Employment length in months"
    )

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "recency": 30.0,
                "frequency": 15,
                "monetary": 5000.0,
                "age": 35,
                "income": 4500.0,
                "employment_length": 24,
            }
        }


class RiskPredictionResponse(BaseModel):
    """
    Risk prediction response model
    """

    customer_id: str
    risk_probability: float = Field(..., description="Probability of default (0-1)")
    credit_score: int = Field(..., description="Credit score (300-850)")
    risk_level: str = Field(..., description="Risk category")
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "risk_probability": 0.15,
                "credit_score": 720,
                "risk_level": "Good",
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class LoanTerms(BaseModel):
    """
    Loan terms model
    """

    max_amount: int = Field(..., description="Maximum loan amount")
    recommended_duration: int = Field(..., description="Recommended duration in months")
    interest_rate: float = Field(..., description="Interest rate")


class RiskAssessment(BaseModel):
    """
    Risk assessment model
    """

    risk_probability: float
    credit_score: int
    risk_level: str


class LoanRecommendationResponse(BaseModel):
    """
    Comprehensive loan recommendation response
    """

    customer_id: str
    risk_assessment: RiskAssessment
    loan_terms: LoanTerms
    recommendation: str = Field(..., description="Loan approval recommendation")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "risk_assessment": {
                    "risk_probability": 0.15,
                    "credit_score": 720,
                    "risk_level": "Good",
                },
                "loan_terms": {
                    "max_amount": 15000,
                    "recommended_duration": 18,
                    "interest_rate": 0.08,
                },
                "recommendation": "APPROVED - Standard terms",
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request model
    """

    batch_id: str = Field(..., description="Unique batch identifier")
    customers: List[CustomerFeatures] = Field(..., description="List of customers")

    class Config:
        schema_extra = {
            "example": {
                "batch_id": "BATCH_001",
                "customers": [
                    {
                        "customer_id": "CUST_001",
                        "recency": 30.0,
                        "frequency": 15,
                        "monetary": 5000.0,
                    }
                ],
            }
        }


class CustomerData(BaseModel):
    """
    Customer data for Task 6 prediction endpoint
    Flexible model that accepts features matching the trained model
    """

    customer_id: str = Field(..., description="Unique customer identifier")

    # Core RFM features (from Task 4)
    recency: Optional[float] = Field(None, description="Days since last transaction")
    frequency: Optional[int] = Field(None, description="Number of transactions")
    monetary: Optional[float] = Field(None, description="Total transaction amount")

    # Additional engineered features (Task 3)
    total_amount: Optional[float] = Field(None, description="Total transaction amount")
    avg_amount: Optional[float] = Field(None, description="Average transaction amount")
    transaction_count: Optional[int] = Field(
        None, description="Total transaction count"
    )
    amount_std: Optional[float] = Field(
        None, description="Standard deviation of amounts"
    )
    amount_range: Optional[float] = Field(
        None, description="Range of transaction amounts"
    )

    # Temporal features
    hour_sin: Optional[float] = Field(None, description="Hour sine encoding")
    hour_cos: Optional[float] = Field(None, description="Hour cosine encoding")
    day_sin: Optional[float] = Field(None, description="Day sine encoding")
    day_cos: Optional[float] = Field(None, description="Day cosine encoding")

    # Additional customer attributes
    age: Optional[int] = Field(None, description="Customer age")
    income: Optional[float] = Field(None, description="Monthly income")
    employment_length: Optional[int] = Field(
        None, description="Employment length in months"
    )

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "recency": 30.0,
                "frequency": 15,
                "monetary": 5000.0,
                "total_amount": 5000.0,
                "avg_amount": 333.33,
                "transaction_count": 15,
                "amount_std": 125.5,
                "age": 35,
                "income": 4500.0,
            }
        }


class PredictionResponse(BaseModel):
    """
    Task 6 prediction response - simple risk probability
    """

    customer_id: str
    risk_probability: float = Field(
        ..., description="Probability of being high risk (0-1)"
    )
    risk_category: str = Field(..., description="Risk category: Low, Medium, High")
    model_info: Dict = Field(..., description="Information about the model used")
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "risk_probability": 0.25,
                "risk_category": "Low",
                "model_info": {"model_type": "XGBoost", "source": "MLflow Registry"},
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Enhanced health check response for Task 6
    """

    status: str = Field(..., description="Service status")
    timestamp: datetime
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict = Field(default_factory=dict, description="Model information")
    version: str = Field(..., description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "model_loaded": True,
                "model_info": {
                    "model_name": "credit-risk-model",
                    "source": "MLflow Registry",
                },
                "version": "1.0.0",
            }
        }
