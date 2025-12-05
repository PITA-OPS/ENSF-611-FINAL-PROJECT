# ENSF-611-FINAL-PROJECT
Term project for ENSF 611, completed by Group 5


# Telco Customer Churn Prediction — ENSF 611 Final Project

Predicting telecom customer churn using machine learning models, including Logistic Regression, Random Forest, and SVM.  
This project follows a complete ML workflow: data loading, cleaning, preprocessing, modeling, hyperparameter tuning, evaluation, and interpretation.

## Project Overview

Customer churn is a major business problem for telecom providers. Retaining a customer is far cheaper than acquiring a new one.  
This project builds a predictive ML pipeline that identifies customers who are most likely to churn based on their service usage, account details, and demographic information.

**Client:** FlexTel Mobile (fictional)  
**Goal:** Predict whether a customer will churn using tabular features.

---

## Dataset

Source: IBM Sample Data 
Raw CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Each record includes:

- Demographics  
- Account information  
- Service subscriptions  
- Billing details  
- Churn label (Yes/No)

---

## Technologies Used

- Python 3  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Google Colab / Jupyter Notebook  

---

## How to Run the Project

### ▶️ **Option 1 — Run in Google Colab (Recommended)**

1. Upload the notebook
2. Run all cells.  
Dataset loads automatically from the GitHub URL.

---

## Data Preprocessing

The project uses **ColumnTransformer** to handle mixed data types:

### Numeric Features  
- Standardized using `StandardScaler()`

### Categorical Features  
- One-hot encoded using `OneHotEncoder(handle_unknown='ignore')`

All preprocessing is included **inside model pipelines** to avoid leakage.

---

## Models Implemented

1. **Logistic Regression** (baseline)  
2. **Random Forest** (baseline + tuned)  
3. **Support Vector Machine (RBF)**  

All models share the same preprocessing pipeline.

---

## Hyperparameter Tuning (Random Forest)

A light GridSearchCV was applied:

```
n_estimators: [100, 200]
max_depth: [None, 10]
min_samples_split: [2, 5]
```

Scored using **ROC-AUC** with 3-fold cross-validation.

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  
- ROC Curves  

These metrics allow comparison across models and highlight class imbalance behavior.

---

## Key Results (Example Structure)

| Model                    | Accuracy | Recall | F1-score | ROC-AUC |
|--------------------------|----------|--------|----------|---------|
| Logistic Regression      | xx       | xx     | xx       | xx      |
| Random Forest (Baseline) | xx       | xx     | xx       | xx      |
| SVM (RBF)                | xx       | xx     | xx       | xx      |
| Random Forest (Tuned)    | xx       | xx     | xx       | xx      |

---

## Feature Importance (Random Forest)

Top predictors include:

- Tenure  
- Contract type  
- MonthlyCharges  
- InternetService  

These features strongly influence churn probability.

---

## Visualizations

- ROC Curve comparison  
- Confusion matrices  
- Feature importance bar chart  

---

## Future Improvements

- Try XGBoost or LightGBM  
- Use SHAP values for deeper interpretability  
- Build a deployment-ready API  
- Perform cost-sensitive learning  

---

## Conclusion

This project demonstrates a complete ML workflow for churn prediction, from preprocessing to model evaluation and interpretation.  
It provides actionable insights into customer behavior and highlights the most influential churn drivers.

---

## Acknowledgements

Dataset © IBM  
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
