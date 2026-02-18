# Customer Churn Prediction  
### End-to-End Machine Learning System with Deployment

This project aims to:

- Predict whether a customer will churn  
- Identify the most influential churn drivers  
- Provide actionable business insights  
- Deploy the model as a real-time prediction service  

This repository demonstrates a complete ML workflow from feature engineering to deployment.

---

## Dataset

Dataset Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains:

- `tenure`
- `monthlycharges`
- `totalcharges`
- `contract`
- `paymentmethod`

Target Variable:
- `Churn` (1 = churn, 0 = retained)

The dataset is imbalanced, requiring careful evaluation.

---

## Feature Engineering (SQL-Based Logic)

Domain-driven feature engineering was applied:

- `is_new_customer` → tenure-based indicator  
- `avg_monthly_spend` → totalcharges / tenure  
- `long_contract` → long-term contract flag  
- `monthly_to_total_ratio` → spending consistency metric  

These engineered features improved both model performance and interpretability.

---

## Model Comparison

Multiple models were evaluated:

| Model | ROC-AUC | F1 Score |
| Logistic Regression | 0.835 | 0.54 |
| Random Forest | 0.811 | 0.53 |
| Logistic (class_weight='balanced') | 0.835 | **0.62** |

### Final Model: Weighted Logistic Regression

Handling class imbalance significantly improved minority class detection:

- ROC-AUC remained stable (~0.835)
- F1 score improved from **0.54 → 0.62**

This model was selected for deployment.

---

## Model Explainability (SHAP)

SHAP analysis identified key churn drivers:

- Contract type  
- Tenure  
- Monthly charges  
- Spending consistency  

### Business Insights

- Short-term contracts increase churn risk  
- Higher monthly charges increase churn probability  
- Long-term contracts reduce churn 

The model is both accurate and interpretable.

---

## Deployment

The final model was:

- Exported using `joblib`
- Deployed via **FastAPI**
- Exposed through a `POST /predict` endpoint
- Integrated with a Bootstrap-based web interface

Users can input customer attributes and receive:

- Binary churn prediction  
- Churn probability score  

---

## Future Improvements
 
- Docker containerization  
- Cloud deployment (AWS / Render / Railway)  
- Monitoring & logging  
- A/B testing simulation  

---

## Project Summary

- SQL-style feature engineering  
- Class imbalance handling  
- Model comparison & optimization  
- SHAP explainability  
- Interactive web interface  

---

### End-to-End ML System  
From business problem → data processing → modeling → explainability → deployment.
