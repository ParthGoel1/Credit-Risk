# Credit Risk Classification

A machine learning project that predicts whether a loan applicant will default, built using real-world credit risk methodology including Weight of Evidence (WoE), Information Value (IV), decile analysis, and KS statistic evaluation.

---

## The Problem

Given customer demographics, loan details, and bureau data, predict whether a customer will default on their loan. The dataset had a significant class imbalance — approximately 45,000 non-default cases against 4,200 default cases — requiring careful handling to avoid a model that simply predicts the majority class.

---

## Datasets

Three datasets merged for this project:

| Dataset | Key Fields |
|---|---|
| Customer data | Age, gender, marital status, employment status, income, dependants, residence type, years at address, city, state, zip code |
| Loan data | Loan purpose, type (secured/unsecured), sanction amount, loan amount, processing fee, GST, net disbursement, tenure, principal outstanding, default (target) |
| Bureau data | Open/closed accounts, total loan months, delinquent months, total days past due, enquiry count, credit utilization ratio |

---

## Data Preprocessing

- Train-test split performed **before** any cleaning or imputation to prevent data leakage
- Missing values populated using mean/mode based on column context
- Identified and handled cases where processing fees exceeded 3% of loan amount
- Engineered financial ratios from raw variables; original raw columns dropped after ratio creation
- Dropped irrelevant features based on EDA insights

---

## Feature Selection

Two methods used depending on variable type:

- **Numerical variables**: Variance Inflation Factor (VIF) to identify and remove multicollinear features
- **Categorical variables**: Weight of Evidence (WoE) and Information Value (IV) to assess predictive power — a standard approach in credit risk modelling

---

## Modelling

Four training runs to systematically address class imbalance:

| Attempt | Approach |
|---|---|
| 1 | Baseline — no class imbalance handling |
| 2 | Undersampling |
| 3 | Oversampling (SMOTE) with Optuna hyperparameter tuning |
| 4 | SMOTE + XGBoost with Optuna hyperparameter tuning |

---

## Results

Evaluated using credit risk industry-standard metrics:

| Metric | Score |
|---|---|
| ROC-AUC | 0.98 |
| Gini Coefficient | 0.96 |
| KS Statistic (top 3 deciles) | 85.98 |

The KS statistic measures the model's ability to separate defaulters from non-defaulters across deciles. A KS of 85.98 in the top three deciles indicates strong discriminatory power where it matters most — the highest-risk segment of the portfolio.

---

## Project Structure

```
credit-risk-classification/
├── data/
│   ├── customers.csv
│   ├── loans.csv
│   └── bureau.csv
├── notebooks/
│   └── credit_risk_classification.ipynb
├── models/
│   └── credit_risk_model.joblib
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn — EDA
- Scikit-learn — Logistic Regression, preprocessing, evaluation
- XGBoost — gradient boosted classification
- Imbalanced-learn — SMOTE oversampling
- Optuna — hyperparameter tuning
- Joblib — model serialisation

---

## Acknowledgements

Dataset and course structure from Codebasics
