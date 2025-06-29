"""
FastAPI Application for Credit Risk Model

This module provides REST API endpoints for:
- Risk probability prediction using MLflow registered model
- Model health checks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Dict
import logging
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

from .pydantic_models import CustomerData, PredictionResponse, HealthCheckResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Model API - Task 6",
    description="API for credit risk prediction using MLflow registered model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
model_info = {}


def load_best_model_from_mlflow():
    """
    Load the best model from MLflow Model Registry
    """
    try:
        # Set MLflow tracking URI (default to local if not set)
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        # Try to load from model registry first
        model_name = os.getenv("MODEL_NAME", "credit-risk-model")
        model_stage = os.getenv("MODEL_STAGE", "Production")

        try:
            # Load from model registry
            model_uri = f"models:/{model_name}/{model_stage}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from registry: {model_uri}")

            # Get model info
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                model_name, stages=[model_stage]
            )[0]
            model_info_dict = {
                "model_name": model_name,
                "model_version": model_version.version,
                "model_stage": model_stage,
                "source": "MLflow Registry",
            }

            return loaded_model, model_info_dict

        except Exception as registry_error:
            logger.warning(f"Failed to load from registry: {registry_error}")

            # Fallback: Load best model from experiments
            experiment_name = os.getenv("EXPERIMENT_NAME", "credit-risk-experiments")

            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    # Get the default experiment
                    experiment = mlflow.get_experiment("0")

                # Search for the best run based on F1 score
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.test_f1_score DESC"],
                    max_results=1,
                )

                if len(runs) > 0:
                    best_run = runs.iloc[0]
                    run_id = best_run.run_id
                    model_uri = f"runs:/{run_id}/model"

                    loaded_model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Loaded best model from run: {run_id}")

                    model_info_dict = {
                        "run_id": run_id,
                        "f1_score": best_run.get("metrics.test_f1_score", "N/A"),
                        "model_type": best_run.get("params.model_type", "Unknown"),
                        "source": "Best Experiment Run",
                    }

                    return loaded_model, model_info_dict
                else:
                    raise Exception("No trained models found in MLflow")

            except Exception as experiment_error:
                logger.error(f"Failed to load from experiments: {experiment_error}")
                raise Exception(f"Could not load model from MLflow: {experiment_error}")

    except Exception as e:
        logger.error(f"MLflow model loading error: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    global model, model_info

    try:
        model, model_info = load_best_model_from_mlflow()
        logger.info("Model loaded successfully from MLflow")
        logger.info(f"Model info: {model_info}")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        model = None
        model_info = {"error": str(e)}


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Credit Risk Model API - Task 6",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    model_loaded = model is not None

    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        model_info=model_info,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Predict risk probability for a customer

    This endpoint accepts customer data matching the model's features
    and returns the risk probability as specified in Task 6.
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Check /health endpoint for details.",
            )

        # Convert customer data to DataFrame
        customer_dict = customer.dict(exclude_unset=True)
        customer_df = pd.DataFrame([customer_dict])

        # Make prediction
        if hasattr(model, "predict_proba"):
            # Get probability of positive class (high risk)
            risk_prob = model.predict_proba(customer_df)[0][1]
        else:
            # For models without predict_proba, use predict and convert
            prediction = model.predict(customer_df)[0]
            risk_prob = (
                float(prediction) if isinstance(prediction, (int, float)) else 0.5
            )

        # Get risk category
        if risk_prob < 0.3:
            risk_category = "Low"
        elif risk_prob < 0.7:
            risk_category = "Medium"
        else:
            risk_category = "High"

        return PredictionResponse(
            customer_id=customer.customer_id,
            risk_probability=float(risk_prob),
            risk_category=risk_category,
            model_info=model_info,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get feature names if available
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = feature_names.tolist()

        info = {
            "model_type": type(model).__name__,
            "model_info": model_info,
            "feature_count": len(feature_names) if feature_names else "Unknown",
            "feature_names": feature_names,
            "model_loaded": True,
        }

        return info

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_task6:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
