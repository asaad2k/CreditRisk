# Credit Risk Prediction Model

A machine learning solution for predicting credit risk using Models. This repository contains a trained models, preprocessing pipeline, and a FastAPI backend for real-time predictions.

## Overview

**Model Purpose**: Binary classification to predict credit risk (TARGET: 0 = Low Risk, 1 = High Risk)

**Model Type**: DecisionTreeClassifier-CatBoost-AdaBoost

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


**Last Updated**: January 2026

**Model Version**: 1.0

**Python Version**: 3.11+
