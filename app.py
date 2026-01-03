"""
FastAPI backend for Credit Risk Prediction Model
This API provides a /predict endpoint that accepts raw feature values and returns risk predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using a Decision Tree classifier",
    version="1.0.0"
)

# Enable CORS for Lovable UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models
try:
    feature_engineer = joblib.load("CreditRisk/feature_engineer.joblib")
    model = joblib.load("CreditRisk/decision_tree_model.joblib")
    print("✓ Models loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error loading models: {e}")
    feature_engineer = None
    model = None

# --- Pydantic Models for Request/Response ---

class PredictionRequest(BaseModel):
    """Request schema for the /predict endpoint."""
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Total Income (must be > 0)")
    AMT_CREDIT: float = Field(..., gt=0, description="Credit Amount (must be > 0)")
    AMT_ANNUITY: float = Field(..., ge=0, description="Annuity (must be >= 0)")
    AMT_GOODS_PRICE: float = Field(..., ge=0, description="Goods Price (must be >= 0)")
    REGION_POPULATION_RELATIVE: float = Field(..., ge=0, le=1, description="Region Population Relative (0-1)")
    DAYS_BIRTH: int = Field(..., description="Days Birth (negative integer, e.g., -12000 to -25000)")
    OWN_CAR_AGE: float = Field(..., ge=0, description="Own Car Age (must be >= 0)")
    OCCUPATION_TYPE: str = Field(..., description="Occupation Type (categorical)")
    WEEKDAY_APPR_PROCESS_START: str = Field(..., description="Weekday (MONDAY-SUNDAY)")
    HOUR_APPR_PROCESS_START: int = Field(..., ge=0, le=23, description="Hour (0-23)")
    ORGANIZATION_TYPE: str = Field(..., description="Organization Type (categorical)")
    OBS_30_CNT_SOCIAL_CIRCLE: int = Field(..., ge=0, description="Observations 30 CNT Social Circle")
    DAYS_LAST_PHONE_CHANGE: int = Field(..., description="Days Last Phone Change (negative integer)")
    AMT_REQ_CREDIT_BUREAU_YEAR: int = Field(..., ge=0, description="Amount Requested Credit Bureau Year")
    EXT_SOURCE_2: float = Field(..., ge=0, le=1, description="External Source 2 (0-1)")

class RiskCategory(str):
    """Risk category based on probability."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    predicted_class: int = Field(..., description="Predicted class (0=Low Risk, 1=High Risk)")
    predicted_probability: float = Field(..., description="Probability of class 1 (High Risk)")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    probability_class_0: float = Field(..., description="Probability of class 0 (Low Risk)")
    probability_class_1: float = Field(..., description="Probability of class 1 (High Risk)")
    threshold_info: str = Field(..., description="Threshold information")

# --- Helper Functions ---

def get_risk_category(probability: float, low_threshold: float = 0.30, high_threshold: float = 0.60) -> str:
    """
    Categorize risk based on probability thresholds.
    
    Args:
        probability: Probability of class 1 (High Risk)
        low_threshold: Threshold for Low risk (default: 0.30)
        high_threshold: Threshold for High risk (default: 0.60)
    
    Returns:
        Risk category (Low, Medium, or High)
    """
    if probability < low_threshold:
        return RiskCategory.LOW
    elif probability < high_threshold:
        return RiskCategory.MEDIUM
    else:
        return RiskCategory.HIGH

def create_input_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """
    Create a DataFrame from the request data with the necessary raw features
    for the FeatureEngineer to process.
    
    The FeatureEngineer expects the raw dataset columns, so we need to create
    a DataFrame with all the necessary columns (even if some are not used).
    """
    # Create a minimal DataFrame with the required columns for FeatureEngineer
    # The FeatureEngineer will drop unnecessary columns and create derived features
    
    # We need to provide the raw features that the FeatureEngineer expects
    # The FeatureEngineer's transform method expects columns like:
    # NAME_INCOME_TYPE, DAYS_EMPLOYED, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, etc.
    
    # For this implementation, we'll create a DataFrame with the 15 selected features
    # and the necessary raw features to derive EXT_SOURCE_2_NEW
    
    data = {
        "AMT_INCOME_TOTAL": [request.AMT_INCOME_TOTAL],
        "AMT_CREDIT": [request.AMT_CREDIT],
        "AMT_ANNUITY": [request.AMT_ANNUITY],
        "AMT_GOODS_PRICE": [request.AMT_GOODS_PRICE],
        "REGION_POPULATION_RELATIVE": [request.REGION_POPULATION_RELATIVE],
        "DAYS_BIRTH": [request.DAYS_BIRTH],
        "OWN_CAR_AGE": [request.OWN_CAR_AGE],
        "OCCUPATION_TYPE": [request.OCCUPATION_TYPE],
        "WEEKDAY_APPR_PROCESS_START": [request.WEEKDAY_APPR_PROCESS_START],
        "HOUR_APPR_PROCESS_START": [request.HOUR_APPR_PROCESS_START],
        "ORGANIZATION_TYPE": [request.ORGANIZATION_TYPE],
        "OBS_30_CNT_SOCIAL_CIRCLE": [request.OBS_30_CNT_SOCIAL_CIRCLE],
        "DAYS_LAST_PHONE_CHANGE": [request.DAYS_LAST_PHONE_CHANGE],
        "AMT_REQ_CREDIT_BUREAU_YEAR": [request.AMT_REQ_CREDIT_BUREAU_YEAR],
        "EXT_SOURCE_2": [request.EXT_SOURCE_2],
        # Additional columns required by FeatureEngineer
        "NAME_INCOME_TYPE": ["Unknown"],  # Placeholder
        "DAYS_EMPLOYED": [0],  # Placeholder
        "EXT_SOURCE_1": [np.nan],  # Not used in final model
        "EXT_SOURCE_3": [np.nan],  # Not used in final model
        "NAME_TYPE_SUITE": ["Unknown"],  # Will be filled
        "CODE_GENDER": ["Unknown"],  # Will be filled
        "CNT_FAM_MEMBERS": [1],  # Placeholder
        "SK_ID_CURR": [0],  # Placeholder
        "TARGET": [0],  # Placeholder (not used in inference)
    }
    
    return pd.DataFrame(data)

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a given applicant.
    
    Args:
        request: PredictionRequest with 15 features
    
    Returns:
        PredictionResponse with prediction, probability, and risk category
    """
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create input DataFrame
        X = create_input_dataframe(request)
        
        # Apply feature engineering
        X_processed = feature_engineer.transform(X)
        
        # Select the 15 features used by the model
        selected_features = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "OWN_CAR_AGE", "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "ORGANIZATION_TYPE",
            "OBS_30_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_YEAR",
            "EXT_SOURCE_2_NEW"
        ]
        
        X_final = X_processed[selected_features]
        
        # Make prediction
        predicted_class = model.predict(X_final)[0]
        predicted_proba = model.predict_proba(X_final)[0]
        
        # Extract probabilities
        prob_class_0 = float(predicted_proba[0])
        prob_class_1 = float(predicted_proba[1])
        
        # Determine risk category
        risk_category = get_risk_category(prob_class_1)
        
        # Prepare response
        return PredictionResponse(
            predicted_class=int(predicted_class),
            predicted_probability=prob_class_1,
            risk_category=risk_category,
            probability_class_0=prob_class_0,
            probability_class_1=prob_class_1,
            threshold_info=f"Low Risk: < 0.30, Medium Risk: 0.30-0.60, High Risk: > 0.60"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the model."""
    return {
        "model_type": "Decision Tree Classifier",
        "selected_features": [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "OWN_CAR_AGE", "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "ORGANIZATION_TYPE",
            "OBS_30_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_YEAR",
            "EXT_SOURCE_2_NEW"
        ],
        "output_classes": ["Low Risk (0)", "High Risk (1)"],
        "risk_thresholds": {
            "low": "< 0.30",
            "medium": "0.30 - 0.60",
            "high": "> 0.60"
        }
    }

# --- Main Entry Point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
