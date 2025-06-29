"""
Pydantic Models for Credit Risk API

This module contains Pydantic models for:
- Request validation
- Response serialization
- Data type enforcement
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    employment_length: Optional[int] = Field(None, description="Employment length in months")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "recency": 30.0,
                "frequency": 15,
                "monetary": 5000.0,
                "age": 35,
                "income": 4500.0,
                "employment_length": 24
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
                "timestamp": "2024-01-15T10:30:00"
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
                    "risk_level": "Good"
                },
                "loan_terms": {
                    "max_amount": 15000,
                    "recommended_duration": 18,
                    "interest_rate": 0.08
                },
                "recommendation": "APPROVED - Standard terms"
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
                        "monetary": 5000.0
                    }
                ]
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Health check response model
    """
    status: str = Field(..., description="Service status")
    timestamp: datetime
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "model_loaded": True,
                "version": "1.0.0"
            }
        } 