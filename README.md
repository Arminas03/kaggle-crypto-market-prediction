# Crypto Market Prediction: Kaggle Competition

## Feature selection and engineering

- LightGBM SHAP values
- Mutual Information regression
- XGBoost gain
- Engineering using field knowledge

## Final Model Architecture

Model | Variants Used | Weight
-|-|-
XGBoost | Underfit + Balanced + Overfit | 20%
LightGBM | Underfit + Balanced + Overfit | 40%
Ridge | Linear regression | 40%


Each model is trained on three data subsets to capture different temporal patterns:
- Last **40%** of the data (short-term): 45% weight
- Last **70%** of the data (mid-term): 45% weight
- **Full** available data (long-term): 10% weight