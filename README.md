# Credit Risk Prediction Model

A machine learning solution for predicting credit risk using a Decision Tree classifier. This repository contains a trained model, preprocessing pipeline, and a FastAPI backend for real-time predictions.

## Overview

**Model Purpose**: Binary classification to predict credit risk (TARGET: 0 = Low Risk, 1 = High Risk)

**Model Type**: Decision Tree Classifier

**Training Data**: Home Credit-style dataset with applicant/employee attributes

**Output**:
- **Predicted Class**: 0 (Low Risk) or 1 (High Risk)
- **Probability Score**: Probability of High Risk (0-1)
- **Risk Category**: Low, Medium, or High (based on configurable thresholds)

## Project Structure

```
CreditRisk/
├── Dataset2.csv                          # Training dataset
├── train_pipeline.py                     # Training script with preprocessing pipeline
├── feature_engineer.joblib               # Fitted feature engineering transformer
├── decision_tree_model.joblib            # Trained Decision Tree model
├── app.py                                # FastAPI backend for inference
├── requirements.txt                      # Python dependencies
├── Model_Pipeline.ipynb                  # Original Jupyter notebook
└── README.md                             # This file
```

## Selected Features (15)

The model uses exactly 15 features for prediction:

### Numeric Features
1. **AMT_INCOME_TOTAL** - Total Income (currency, > 0)
2. **AMT_CREDIT** - Credit Amount (currency, > 0)
3. **AMT_ANNUITY** - Annuity (currency, >= 0)
4. **AMT_GOODS_PRICE** - Goods Price (currency, >= 0)
5. **REGION_POPULATION_RELATIVE** - Region Population Relative (0-1, decimal)
6. **DAYS_BIRTH** - Days Birth (negative integer, e.g., -12000 to -25000)
7. **OWN_CAR_AGE** - Own Car Age (years, >= 0)
8. **HOUR_APPR_PROCESS_START** - Application Hour (0-23)
9. **OBS_30_CNT_SOCIAL_CIRCLE** - Observations 30 CNT Social Circle (integer, >= 0)
10. **DAYS_LAST_PHONE_CHANGE** - Days Last Phone Change (negative integer)
11. **AMT_REQ_CREDIT_BUREAU_YEAR** - Amount Requested Credit Bureau Year (integer, >= 0)
12. **REGION_POPULATION_RELATIVE** - Region Population Relative (0-1)

### Categorical Features
13. **OCCUPATION_TYPE** - Occupation Type (categorical, encoded)
14. **WEEKDAY_APPR_PROCESS_START** - Weekday of Application (MONDAY-SUNDAY)
15. **ORGANIZATION_TYPE** - Organization Type (categorical, encoded)

### Derived Features
- **EXT_SOURCE_2_NEW** - Derived from EXT_SOURCE_2 (raw input):
  - "dont have record" if missing
  - "Below 0.5" if < 0.5
  - "Over 0.5" if >= 0.5

## Preprocessing Pipeline

The preprocessing pipeline applies the following transformations:

### 1. Data Cleaning
- Drop irrelevant columns (e.g., DAYS_REGISTRATION, APARTMENTS_AVG, etc.)
- Replace "XNA" with NaN
- Convert object columns to numeric where possible
- Apply absolute values to numeric columns

### 2. Feature Engineering
- **DAYS_EMPLOYED_NEW**: Derived from DAYS_EMPLOYED and NAME_INCOME_TYPE
  - "N" if Unemployed or DAYS_EMPLOYED > 180
  - "Y" if Pensioner or employed
- **EXT_SOURCE_*_NEW**: Categorical encoding of external source scores
- Missing value imputation (median for numeric, mode for categorical)

### 3. Encoding
- Label Encoding for categorical features (fitted during training)
- Preserves encoding consistency between training and inference

### 4. Transformations
- Log transformation (log1p) for skewed numeric features
- MinMax scaling (0-1) for numeric features

## API Documentation

### Endpoints

#### 1. Health Check
```
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_engineer_loaded": true
}
```

#### 2. Prediction
```
POST /predict
```

**Request Body** (JSON):
```json
{
  "AMT_INCOME_TOTAL": 100000,
  "AMT_CREDIT": 50000,
  "AMT_ANNUITY": 5000,
  "AMT_GOODS_PRICE": 48000,
  "REGION_POPULATION_RELATIVE": 0.015,
  "DAYS_BIRTH": -15000,
  "OWN_CAR_AGE": 5,
  "OCCUPATION_TYPE": "Laborers",
  "WEEKDAY_APPR_PROCESS_START": "MONDAY",
  "HOUR_APPR_PROCESS_START": 10,
  "ORGANIZATION_TYPE": "Business Entity Type 3",
  "OBS_30_CNT_SOCIAL_CIRCLE": 2,
  "DAYS_LAST_PHONE_CHANGE": -1000,
  "AMT_REQ_CREDIT_BUREAU_YEAR": 1,
  "EXT_SOURCE_2": 0.65
}
```

**Response** (JSON):
```json
{
  "predicted_class": 0,
  "predicted_probability": 0.25,
  "risk_category": "Low",
  "probability_class_0": 0.75,
  "probability_class_1": 0.25,
  "threshold_info": "Low Risk: < 0.30, Medium Risk: 0.30-0.60, High Risk: > 0.60"
}
```

**Response Fields**:
- `predicted_class`: 0 (Low Risk) or 1 (High Risk)
- `predicted_probability`: Probability of High Risk (0-1)
- `risk_category`: Categorical risk level (Low/Medium/High)
- `probability_class_0`: Probability of Low Risk
- `probability_class_1`: Probability of High Risk
- `threshold_info`: Risk category thresholds

#### 3. Model Information
```
GET /model-info
```

**Response**:
```json
{
  "model_type": "Decision Tree Classifier",
  "selected_features": [...],
  "output_classes": ["Low Risk (0)", "High Risk (1)"],
  "risk_thresholds": {
    "low": "< 0.30",
    "medium": "0.30 - 0.60",
    "high": "> 0.60"
  }
}
```

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/asaad2k/CreditRisk.git
cd CreditRisk
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API Server
```bash
python app.py
```

The API will be available at `http://localhost:8000`

**Interactive API Documentation**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment Options

### Option A: Local Testing with FastAPI + Uvicorn
```bash
source venv/bin/activate
python app.py
```

### Option B: Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t creditrisk-api .
docker run -p 8000:8000 creditrisk-api
```

### Option C: Cloud Deployment
Deploy to:
- **Google Cloud Run**: `gcloud run deploy creditrisk-api --source .`
- **Heroku**: `git push heroku main`
- **Railway**: Connect GitHub repo and deploy
- **Render**: Connect GitHub repo and deploy
- **HuggingFace Spaces**: Upload files and configure

### Option D: Temporary Public Access (Google Colab)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok

# Setup ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# Run FastAPI with ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Start uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## UI Integration (Lovable)

### Form Fields to Collect

| Field | Type | Validation | Example |
|-------|------|-----------|---------|
| Total Income | Number | > 0 | 100000 |
| Credit Amount | Number | > 0 | 50000 |
| Annuity | Number | >= 0 | 5000 |
| Goods Price | Number | >= 0 | 48000 |
| Region Population Relative | Number | 0-1 | 0.015 |
| Days Birth | Number | Negative int | -15000 |
| Own Car Age | Number | >= 0 | 5 |
| Occupation Type | Dropdown | Categorical | Laborers |
| Weekday | Dropdown | MONDAY-SUNDAY | MONDAY |
| Hour | Number | 0-23 | 10 |
| Organization Type | Dropdown | Categorical | Business Entity Type 3 |
| Social Circle (30 days) | Number | >= 0 | 2 |
| Days Last Phone Change | Number | Negative int | -1000 |
| Credit Bureau Requests (Year) | Number | >= 0 | 1 |
| External Source 2 | Number | 0-1 | 0.65 |

### JavaScript/React Integration Example

```javascript
async function predictRisk(formData) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  });
  
  const result = await response.json();
  
  // Display results
  console.log(`Risk Category: ${result.risk_category}`);
  console.log(`Probability: ${(result.predicted_probability * 100).toFixed(2)}%`);
  
  return result;
}
```

## Risk Category Interpretation

| Category | Probability Range | Interpretation |
|----------|-------------------|-----------------|
| **Low** | < 0.30 (30%) | Low likelihood of default; safe to approve |
| **Medium** | 0.30 - 0.60 (30-60%) | Moderate risk; may require additional review |
| **High** | > 0.60 (60%) | High likelihood of default; recommend rejection |

**Note**: Risk thresholds are configurable and should be adjusted based on business requirements.

## Model Performance

The Decision Tree model was trained and evaluated using:
- **Train/Test Split**: 80/20 stratified split
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Metrics**: Accuracy, ROC-AUC, Confusion Matrix
- **Feature Selection**: RFE (Recursive Feature Elimination) to select 15 optimal features

## Training & Retraining

To retrain the model with new data:

```bash
# Update Dataset2.csv with new data
python train_pipeline.py
```

This will:
1. Load and balance the dataset
2. Apply feature engineering and preprocessing
3. Train the Decision Tree model
4. Save the updated `feature_engineer.joblib` and `decision_tree_model.joblib`

## Important Notes

### Data Format Requirements
- **DAYS_BIRTH** and **DAYS_LAST_PHONE_CHANGE** must be negative integers (days before application date)
- **REGION_POPULATION_RELATIVE** and **EXT_SOURCE_2** must be between 0 and 1
- **WEEKDAY_APPR_PROCESS_START** must match exact spelling: MONDAY, TUESDAY, ..., SUNDAY
- **OCCUPATION_TYPE** and **ORGANIZATION_TYPE** must be from the training dataset categories

### Preprocessing Consistency
- The inference pipeline applies the **exact same preprocessing** as training
- Missing values are handled according to training logic
- Categorical encoding uses fitted LabelEncoders from training
- Numeric scaling uses fitted MinMaxScaler from training

### Handling Unseen Categories
If a categorical value (OCCUPATION_TYPE or ORGANIZATION_TYPE) is not in the training data:
1. The LabelEncoder will raise an error
2. Recommendation: Provide a dropdown with only training categories
3. Alternative: Map unknown categories to "Unknown" or "Other" (if present in training)

## Troubleshooting

### Issue: "Model not loaded"
**Solution**: Ensure `feature_engineer.joblib` and `decision_tree_model.joblib` exist in the `CreditRisk/` directory.

### Issue: "Prediction failed: KeyError"
**Solution**: Check that all 15 required features are provided in the request.

### Issue: "Unknown category in OCCUPATION_TYPE"
**Solution**: Provide only categories that exist in the training dataset. See `feature_engineer.joblib` for valid categories.

### Issue: CORS errors when calling from Lovable
**Solution**: Ensure the API is running with CORS enabled (default in `app.py`).

## File Descriptions

### `train_pipeline.py`
- Loads and preprocesses the training dataset
- Applies feature engineering (DAYS_EMPLOYED_NEW, EXT_SOURCE_*_NEW)
- Trains the Decision Tree model
- Saves the FeatureEngineer and model as joblib files

### `feature_engineer.joblib`
- Fitted transformer containing:
  - LabelEncoders for categorical features
  - MinMaxScaler for numeric features
  - Log transformation configuration
  - Missing value handling logic

### `decision_tree_model.joblib`
- Trained Decision Tree classifier
- Ready for predictions on preprocessed data

### `app.py`
- FastAPI application with /predict endpoint
- Loads models and handles inference
- Applies preprocessing to raw input
- Returns predictions with probabilities and risk categories

## Contact & Support

For issues or questions, please open an issue on GitHub or contact the project maintainer.

---

**Last Updated**: January 2026

**Model Version**: 1.0

**Python Version**: 3.11+
